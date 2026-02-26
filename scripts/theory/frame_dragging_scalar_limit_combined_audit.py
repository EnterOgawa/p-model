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


# 関数: `_to_float` の入出力契約と処理意図を定義する。

def _to_float(v: Any) -> Optional[float]:
    try:
        val = float(v)
    except Exception:
        return None

    # 条件分岐: `math.isnan(val) or math.isinf(val)` を満たす経路を評価する。

    if math.isnan(val) or math.isinf(val):
        return None

    return val


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(gpb_payload: Dict[str, Any], frame_payload: Dict[str, Any], z_gate: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    gpb_channels = gpb_payload.get("channels") or []
    geodetic = None
    frame_dragging = None
    for ch in gpb_channels:
        cid = str(ch.get("id") or "")
        # 条件分岐: `cid == "geodetic_precession"` を満たす経路を評価する。
        if cid == "geodetic_precession":
            geodetic = ch
        # 条件分岐: 前段条件が不成立で、`cid == "frame_dragging"` を追加評価する。
        elif cid == "frame_dragging":
            frame_dragging = ch

    # 条件分岐: `geodetic is not None` を満たす経路を評価する。

    if geodetic is not None:
        observed = _to_float(geodetic.get("observed"))
        sigma = _to_float(geodetic.get("observed_sigma"))
        pred_scalar = _to_float(geodetic.get("pmodel_scalar_prediction"))
        residual = None
        z_scalar = None
        status = "inconclusive"
        # 条件分岐: `observed is not None and pred_scalar is not None` を満たす経路を評価する。
        if observed is not None and pred_scalar is not None:
            residual = observed - pred_scalar
            # 条件分岐: `sigma is not None and sigma > 0` を満たす経路を評価する。
            if sigma is not None and sigma > 0:
                z_scalar = residual / sigma
                status = "reject" if abs(z_scalar) > z_gate else "pass"

        rows.append(
            {
                "id": "gpb_geodetic_control",
                "experiment": "GP-B",
                "observable": "geodetic_precession",
                "label": "GP-B geodetic (control)",
                "value_domain": "absolute",
                "unit": str(geodetic.get("unit") or "arcsec/yr"),
                "observed": observed,
                "observed_sigma": sigma,
                "pmodel_scalar_prediction": pred_scalar,
                "reference_prediction": _to_float(geodetic.get("reference_prediction")),
                "residual_scalar": residual,
                "z_scalar": z_scalar,
                "status": status,
                "note": str(geodetic.get("note") or ""),
            }
        )

    # 条件分岐: `frame_dragging is not None` を満たす経路を評価する。

    if frame_dragging is not None:
        observed = _to_float(frame_dragging.get("observed"))
        sigma = _to_float(frame_dragging.get("observed_sigma"))
        pred_scalar = _to_float(frame_dragging.get("pmodel_scalar_prediction"))
        residual = None
        z_scalar = None
        status = "inconclusive"
        # 条件分岐: `observed is not None and pred_scalar is not None` を満たす経路を評価する。
        if observed is not None and pred_scalar is not None:
            residual = observed - pred_scalar
            # 条件分岐: `sigma is not None and sigma > 0` を満たす経路を評価する。
            if sigma is not None and sigma > 0:
                z_scalar = residual / sigma
                status = "reject" if abs(z_scalar) > z_gate else "pass"

        rows.append(
            {
                "id": "gpb_frame_dragging_scalar_limit",
                "experiment": "GP-B",
                "observable": "frame_dragging",
                "label": "GP-B frame-dragging",
                "value_domain": "absolute",
                "unit": str(frame_dragging.get("unit") or "arcsec/yr"),
                "observed": observed,
                "observed_sigma": sigma,
                "pmodel_scalar_prediction": pred_scalar,
                "reference_prediction": _to_float(frame_dragging.get("reference_prediction")),
                "residual_scalar": residual,
                "z_scalar": z_scalar,
                "status": status,
                "note": str(frame_dragging.get("note") or ""),
            }
        )

    experiments = frame_payload.get("experiments") or []
    for exp in experiments:
        exp_id = str(exp.get("id") or "")
        # 条件分岐: `"lageos" not in exp_id.lower()` を満たす経路を評価する。
        if "lageos" not in exp_id.lower():
            continue

        observed_mu = _to_float(exp.get("mu"))
        sigma_mu = _to_float(exp.get("mu_sigma"))
        # 条件分岐: `observed_mu is None` を満たす経路を評価する。
        if observed_mu is None:
            obs_rate = _to_float(exp.get("omega_obs_mas_per_yr"))
            pred_rate = _to_float(exp.get("omega_pred_mas_per_yr"))
            # 条件分岐: `obs_rate is not None and pred_rate is not None and abs(pred_rate) > 0` を満たす経路を評価する。
            if obs_rate is not None and pred_rate is not None and abs(pred_rate) > 0:
                observed_mu = abs(obs_rate) / abs(pred_rate)
                sigma_rate = _to_float(exp.get("omega_obs_sigma_mas_per_yr"))
                # 条件分岐: `sigma_rate is not None and abs(pred_rate) > 0` を満たす経路を評価する。
                if sigma_rate is not None and abs(pred_rate) > 0:
                    sigma_mu = abs(sigma_rate) / abs(pred_rate)

        pred_scalar = 0.0
        residual = None
        z_scalar = None
        status = "inconclusive"
        # 条件分岐: `observed_mu is not None` を満たす経路を評価する。
        if observed_mu is not None:
            residual = observed_mu - pred_scalar
            # 条件分岐: `sigma_mu is not None and sigma_mu > 0` を満たす経路を評価する。
            if sigma_mu is not None and sigma_mu > 0:
                z_scalar = residual / sigma_mu
                status = "reject" if abs(z_scalar) > z_gate else "pass"

        rows.append(
            {
                "id": "lageos_frame_dragging_scalar_limit",
                "experiment": "LAGEOS",
                "observable": "frame_dragging",
                "label": "LAGEOS frame-dragging",
                "value_domain": "ratio_to_gr",
                "unit": "mu_ratio",
                "observed": observed_mu,
                "observed_sigma": sigma_mu,
                "pmodel_scalar_prediction": pred_scalar,
                "reference_prediction": 1.0,
                "residual_scalar": residual,
                "z_scalar": z_scalar,
                "status": status,
                "note": str(exp.get("sigma_note") or ""),
            }
        )
        break

    return rows


# 関数: `_build_summary` の入出力契約と処理意図を定義する。

def _build_summary(rows: Sequence[Dict[str, Any]], z_gate: float) -> Dict[str, Any]:
    n_pass = sum(1 for r in rows if r.get("status") == "pass")
    n_reject = sum(1 for r in rows if r.get("status") == "reject")
    n_inconclusive = sum(1 for r in rows if r.get("status") == "inconclusive")

    frame_rows = [r for r in rows if str(r.get("observable") or "") == "frame_dragging"]
    frame_reject_rows = [r for r in frame_rows if r.get("status") == "reject"]
    frame_reject_experiments = sorted({str(r.get("experiment") or "") for r in frame_reject_rows})

    overall = "inconclusive"
    # 条件分岐: `frame_reject_rows` を満たす経路を評価する。
    if frame_reject_rows:
        overall = "reject"
    # 条件分岐: 前段条件が不成立で、`frame_rows and all(r.get("status") == "pass" for r in frame_rows)` を追加評価する。
    elif frame_rows and all(r.get("status") == "pass" for r in frame_rows):
        overall = "pass"

    return {
        "channels_n": len(rows),
        "frame_dragging_channels_n": len(frame_rows),
        "pass_n": n_pass,
        "reject_n": n_reject,
        "inconclusive_n": n_inconclusive,
        "frame_dragging_reject_n": len(frame_reject_rows),
        "frame_dragging_reject_experiments": frame_reject_experiments,
        "multi_experiment_reject_confirmed": len(frame_reject_experiments) >= 2,
        "gate_z_reject": z_gate,
        "overall_status": overall,
        "decision": (
            "scalar_limit_rejected_by_frame_dragging"
            if overall == "reject"
            else "scalar_limit_not_rejected_by_current_gate"
        ),
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(rows: Sequence[Dict[str, Any]], z_gate: float, out_png: Path) -> None:
    _set_japanese_font()

    labels = [str(r.get("label") or r.get("id") or "") for r in rows]
    zvals = np.array([float(r["z_scalar"]) if r.get("z_scalar") is not None else np.nan for r in rows], dtype=float)
    x = np.arange(len(rows), dtype=float)
    colors = []
    for r, z in zip(rows, zvals):
        # 条件分岐: `r.get("status") == "reject"` を満たす経路を評価する。
        if r.get("status") == "reject":
            colors.append("#d62728")
        # 条件分岐: 前段条件が不成立で、`r.get("status") == "pass"` を追加評価する。
        elif r.get("status") == "pass":
            colors.append("#2ca02c")
        # 条件分岐: 前段条件が不成立で、`math.isfinite(float(z))` を追加評価する。
        elif math.isfinite(float(z)):
            colors.append("#bcbd22")
        else:
            colors.append("#7f7f7f")

    fig = plt.figure(figsize=(10.8, 5.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(x, np.nan_to_num(zvals, nan=0.0), color=colors, alpha=0.92)
    ax.axvline(z_gate, color="#333333", linestyle="--", linewidth=1.1)
    ax.axvline(-z_gate, color="#333333", linestyle="--", linewidth=1.1)
    ax.axvline(0.0, color="#666666", linestyle="-", linewidth=0.9, alpha=0.8)

    for yi, zi in zip(x, zvals):
        # 条件分岐: `math.isfinite(float(zi))` を満たす経路を評価する。
        if math.isfinite(float(zi)):
            pad = 0.1 if zi >= 0 else -0.1
            ha = "left" if zi >= 0 else "right"
            ax.text(float(zi) + pad, yi, _fmt_float(float(zi), 3), va="center", ha=ha, fontsize=9)

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.set_xlabel("z = (obs - pred_scalar) / sigma_obs")
    ax.set_title("Scalar-limit gate (GP-B + LAGEOS)")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "experiment",
                "observable",
                "label",
                "value_domain",
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
                    r.get("experiment", ""),
                    r.get("observable", ""),
                    r.get("label", ""),
                    r.get("value_domain", ""),
                    r.get("unit", ""),
                    "" if r.get("observed") is None else _fmt_float(float(r["observed"]), 6),
                    "" if r.get("observed_sigma") is None else _fmt_float(float(r["observed_sigma"]), 6),
                    ""
                    if r.get("pmodel_scalar_prediction") is None
                    else _fmt_float(float(r["pmodel_scalar_prediction"]), 6),
                    ""
                    if r.get("reference_prediction") is None
                    else _fmt_float(float(r["reference_prediction"]), 6),
                    "" if r.get("residual_scalar") is None else _fmt_float(float(r["residual_scalar"]), 6),
                    "" if r.get("z_scalar") is None else _fmt_float(float(r["z_scalar"]), 6),
                    r.get("status", ""),
                    r.get("note", ""),
                ]
            )


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _ROOT
    default_gpb_data = root / "data" / "theory" / "gpb_scalar_limit_audit.json"
    default_frame_data = root / "data" / "theory" / "frame_dragging_experiments.json"
    default_outdir = root / "output" / "private" / "theory"
    default_public_outdir = root / "output" / "public" / "theory"

    ap = argparse.ArgumentParser(description="Combined scalar-limit gate using GP-B + LAGEOS frame-dragging observables.")
    ap.add_argument("--gpb-data", type=str, default=str(default_gpb_data), help="Input JSON for GP-B scalar audit channels.")
    ap.add_argument("--frame-data", type=str, default=str(default_frame_data), help="Input JSON for frame-dragging experiments (LAGEOS μ).")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory.")
    ap.add_argument("--public-outdir", type=str, default=str(default_public_outdir), help="Public output directory.")
    ap.add_argument("--z-reject", type=float, default=3.0, help="Reject gate on |z|.")
    ap.add_argument("--no-public-copy", action="store_true", help="Do not copy outputs to public directory.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    gpb_data = Path(args.gpb_data)
    frame_data = Path(args.frame_data)
    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)

    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    gpb_payload = _read_json(gpb_data)
    frame_payload = _read_json(frame_data)
    rows = _build_rows(gpb_payload, frame_payload, z_gate=float(args.z_reject))
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        raise SystemExit("no rows generated")

    summary = _build_summary(rows, z_gate=float(args.z_reject))

    out_json = outdir / "frame_dragging_scalar_limit_combined_audit.json"
    out_csv = outdir / "frame_dragging_scalar_limit_combined_audit.csv"
    out_png = outdir / "frame_dragging_scalar_limit_combined_audit.png"

    payload_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "wavep.theory.frame_dragging_scalar_limit_combined_audit.v1",
        "title": "Frame-dragging scalar-limit combined audit (GP-B + LAGEOS)",
        "gate": {"z_reject": float(args.z_reject)},
        "inputs": {
            "gpb_data": str(gpb_data).replace("\\", "/"),
            "frame_data": str(frame_data).replace("\\", "/"),
        },
        "rows": rows,
        "summary": summary,
        "outputs": {
            "rows_json": str(out_json).replace("\\", "/"),
            "rows_csv": str(out_csv).replace("\\", "/"),
            "plot_png": str(out_png).replace("\\", "/"),
        },
    }

    _plot(rows, float(args.z_reject), out_png)
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
                "event_type": "theory_frame_dragging_scalar_limit_combined_audit",
                "argv": sys.argv,
                "inputs": {"gpb_data": gpb_data, "frame_data": frame_data},
                "outputs": {
                    "rows_json": out_json,
                    "rows_csv": out_csv,
                    "plot_png": out_png,
                    "public_copies": copied,
                },
                "metrics": {
                    "channels_n": summary.get("channels_n"),
                    "frame_dragging_channels_n": summary.get("frame_dragging_channels_n"),
                    "frame_dragging_reject_n": summary.get("frame_dragging_reject_n"),
                    "overall_status": summary.get("overall_status"),
                    "multi_experiment_reject_confirmed": summary.get("multi_experiment_reject_confirmed"),
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
