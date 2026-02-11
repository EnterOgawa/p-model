#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weak_field_longterm_consistency.py

Phase 6 / Step 6.2.3:
弱場（太陽系）テストを長期・多系統で「同一I/F」で一括更新し、
主要メトリクスを集約して固定出力する。

このスクリプトは以下を行う：
  1) 既存出力（各テストの metrics / summary）を収集（既定）
  2) （任意）基準パラメータでテストを再実行してから収集（--rerun）

出力（固定）:
  - output/summary/weak_field_longterm_consistency.json
  - output/summary/weak_field_longterm_consistency.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        pass


def _try_load_frozen_parameters() -> Dict[str, Any]:
    p = _ROOT / "output" / "theory" / "frozen_parameters.json"
    if not p.exists():
        return {"path": _relpath(p), "exists": False}
    try:
        data = _read_json(p)
    except Exception:
        return {"path": _relpath(p), "exists": True, "parse_error": True}
    out: Dict[str, Any] = {"path": _relpath(p), "exists": True}
    for k in ("beta", "beta_sigma", "gamma_pmodel", "gamma_pmodel_sigma", "delta"):
        if k in data:
            out[k] = data.get(k)
    policy = data.get("policy")
    if isinstance(policy, dict):
        out["policy"] = {kk: policy.get(kk) for kk in ("fit_predict_separation", "beta_source", "delta_source", "note")}
    return out


def _load_cassini_metrics(csv_path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            window = str(r.get("window") or "").strip()
            if not window:
                continue
            out: Dict[str, float] = {}
            for k in ("n", "rmse", "mae", "corr", "max_obs", "min_obs", "max_model", "min_model"):
                v = r.get(k)
                if v is None or str(v).strip() == "":
                    continue
                try:
                    out[k] = float(v)
                except Exception:
                    continue
            rows[window] = out
    return rows


def _load_gps_metrics(json_path: Path) -> Dict[str, Any]:
    d = _read_json(json_path)
    return d.get("metrics") if isinstance(d.get("metrics"), dict) else {}


def _load_llr_summary(json_path: Path) -> Dict[str, Any]:
    d = _read_json(json_path)
    return d.get("median_rms_ns") if isinstance(d.get("median_rms_ns"), dict) else {}


def _load_mercury_metrics(json_path: Path) -> Dict[str, Any]:
    d = _read_json(json_path)
    ref = d.get("reference_arcsec_century")
    sim = (((d.get("simulation_physical") or {}).get("pmodel") or {}).get("arcsec_per_century"))
    ein = (((d.get("einstein_approx") or {}).get("arcsec_per_century")))
    out: Dict[str, Any] = {}
    if isinstance(ref, (int, float)):
        out["reference_arcsec_century"] = float(ref)
    if isinstance(sim, (int, float)):
        out["pmodel_arcsec_century"] = float(sim)
    if isinstance(ein, (int, float)):
        out["einstein_arcsec_century"] = float(ein)
    if "reference_arcsec_century" in out and "pmodel_arcsec_century" in out:
        out["pmodel_minus_ref_arcsec_century"] = out["pmodel_arcsec_century"] - out["reference_arcsec_century"]
        out["pmodel_minus_ref_percent"] = 100.0 * out["pmodel_minus_ref_arcsec_century"] / out["reference_arcsec_century"]
    return out


def _load_viking_series(csv_path: Path) -> Dict[str, Any]:
    max_us = None
    max_time = None
    n = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = str(r.get("time_utc") or "").strip()
            v = r.get("shapiro_delay_us")
            if v is None or str(v).strip() == "":
                continue
            try:
                us = float(v)
            except Exception:
                continue
            n += 1
            if (max_us is None) or (us > max_us):
                max_us = us
                max_time = t or None
    out: Dict[str, Any] = {"n": int(n)}
    if max_us is not None:
        out["max_delay_us"] = float(max_us)
        out["max_time_utc"] = max_time
    return out


def _run(cmd: List[str], *, cwd: Path, timeout_s: float, log_path: Path) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n--- run start: {datetime.now(timezone.utc).isoformat()} ---\n")
        logf.write("cmd: " + " ".join(cmd) + "\n")
        logf.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "returncode": None, "timeout_s": timeout_s, "elapsed_s": time.perf_counter() - t0}
        finally:
            logf.flush()

    elapsed = time.perf_counter() - t0
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"returncode: {proc.returncode}\n")
        if proc.stdout:
            logf.write("--- stdout ---\n")
            logf.write(proc.stdout[:20000] + ("\n...[truncated]\n" if len(proc.stdout) > 20000 else ""))
        if proc.stderr:
            logf.write("--- stderr ---\n")
            logf.write(proc.stderr[:20000] + ("\n...[truncated]\n" if len(proc.stderr) > 20000 else ""))
        logf.write(f"--- run end (elapsed_s={elapsed:.3f}) ---\n")
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "elapsed_s": elapsed}


def collect(
    *,
    matrix_path: Path,
    templates_path: Path,
    rerun: bool,
    timeout_s: float,
    tests_filter: Optional[List[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    matrix = _read_json(matrix_path)
    templates = _read_json(templates_path)
    frozen = _try_load_frozen_parameters()

    beta = frozen.get("beta")
    delta = frozen.get("delta")
    beta_val = float(beta) if isinstance(beta, (int, float)) else 1.0
    delta_val = float(delta) if isinstance(delta, (int, float)) else 0.0

    # Sources (fixed paths; should be stable outputs from each topic)
    cassini_metrics_csv = _ROOT / "output" / "cassini" / "cassini_fig2_metrics.csv"
    gps_metrics_json = _ROOT / "output" / "gps" / "gps_compare_metrics.json"
    llr_summary_json = _ROOT / "output" / "llr" / "batch" / "llr_batch_summary.json"
    mercury_metrics_json = _ROOT / "output" / "mercury" / "mercury_precession_metrics.json"
    viking_series_csv = _ROOT / "output" / "viking" / "viking_shapiro_result.csv"

    out_dir = _ROOT / "output" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "weak_field_longterm_consistency_run.log"

    rerun_status: Dict[str, Any] = {}
    if rerun:
        planned: List[Tuple[str, List[str]]] = [
            (
                "cassini_sce1_doppler",
                [sys.executable, "-B", str(_ROOT / "scripts" / "cassini" / "cassini_fig2_overlay.py"), "--beta", str(beta_val)],
            ),
            (
                "viking_shapiro_peak",
                [
                    sys.executable,
                    "-B",
                    str(_ROOT / "scripts" / "viking" / "viking_shapiro_check.py"),
                    "--beta",
                    str(beta_val),
                    "--offline",
                ],
            ),
            ("mercury_perihelion_precession", [sys.executable, "-B", str(_ROOT / "scripts" / "mercury" / "mercury_precession_v3.py")]),
            (
                "gps_satellite_clock",
                [sys.executable, "-B", str(_ROOT / "scripts" / "gps" / "compare_clocks.py"), "--delta", str(delta_val)],
            ),
            (
                "llr_batch",
                [
                    sys.executable,
                    "-B",
                    str(_ROOT / "scripts" / "llr" / "llr_batch_eval.py"),
                    "--beta",
                    str(beta_val),
                    "--time-tag-mode",
                    "auto",
                    "--offline",
                    "--chunk",
                    "50",
                ],
            ),
        ]
        for test_id, cmd in planned:
            if tests_filter and test_id not in tests_filter:
                rerun_status[test_id] = {"skipped": True, "reason": "filtered"}
                continue
            rerun_status[test_id] = _run(cmd, cwd=_ROOT, timeout_s=timeout_s, log_path=run_log)

    results: Dict[str, Any] = {}

    def _want(test_id: str) -> bool:
        return (tests_filter is None) or (test_id in tests_filter)

    if _want("cassini_sce1_doppler"):
        if cassini_metrics_csv.exists():
            rows = _load_cassini_metrics(cassini_metrics_csv)
            focus = rows.get("-10 to +10 days") or rows.get("all (available points)") or {}
            results["cassini_sce1_doppler"] = {
                "status": "ok",
                "source": {"cassini_fig2_metrics_csv": _relpath(cassini_metrics_csv)},
                "windows": rows,
                "focus": {"window": "-10 to +10 days", "rmse": focus.get("rmse"), "corr": focus.get("corr")},
            }
        else:
            results["cassini_sce1_doppler"] = {"status": "missing", "source": {"cassini_fig2_metrics_csv": _relpath(cassini_metrics_csv)}}

    if _want("gps_satellite_clock"):
        if gps_metrics_json.exists():
            m = _load_gps_metrics(gps_metrics_json)
            brdc = m.get("brdc_rms_ns_median")
            pmod = m.get("pmodel_rms_ns_median")
            ratio = None
            if isinstance(brdc, (int, float)) and isinstance(pmod, (int, float)) and float(brdc) != 0.0:
                ratio = float(pmod) / float(brdc)
            results["gps_satellite_clock"] = {
                "status": "ok",
                "source": {"gps_compare_metrics_json": _relpath(gps_metrics_json)},
                "metrics": m,
                "derived": {"pmodel_over_brdc_rms_median_ratio": ratio},
            }
        else:
            results["gps_satellite_clock"] = {"status": "missing", "source": {"gps_compare_metrics_json": _relpath(gps_metrics_json)}}

    if _want("llr_batch"):
        if llr_summary_json.exists():
            m = _load_llr_summary(llr_summary_json)
            tide = m.get("station_reflector_tropo_tide")
            nosh = m.get("station_reflector_tropo_no_shapiro")
            ratio = None
            if isinstance(tide, (int, float)) and isinstance(nosh, (int, float)) and float(nosh) != 0.0:
                ratio = float(tide) / float(nosh)
            results["llr_batch"] = {
                "status": "ok",
                "source": {"llr_batch_summary_json": _relpath(llr_summary_json)},
                "median_rms_ns": m,
                "derived": {"tropo_tide_over_tropo_no_shapiro_ratio": ratio},
            }
        else:
            results["llr_batch"] = {"status": "missing", "source": {"llr_batch_summary_json": _relpath(llr_summary_json)}}

    if _want("mercury_perihelion_precession"):
        if mercury_metrics_json.exists():
            m = _load_mercury_metrics(mercury_metrics_json)
            results["mercury_perihelion_precession"] = {
                "status": "ok",
                "source": {"mercury_precession_metrics_json": _relpath(mercury_metrics_json)},
                "metrics": m,
            }
        else:
            results["mercury_perihelion_precession"] = {
                "status": "missing",
                "source": {"mercury_precession_metrics_json": _relpath(mercury_metrics_json)},
            }

    if _want("viking_shapiro_peak"):
        if viking_series_csv.exists():
            m = _load_viking_series(viking_series_csv)
            results["viking_shapiro_peak"] = {
                "status": "ok",
                "source": {"viking_shapiro_result_csv": _relpath(viking_series_csv)},
                "series": m,
            }
        else:
            results["viking_shapiro_peak"] = {"status": "missing", "source": {"viking_shapiro_result_csv": _relpath(viking_series_csv)}}

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 6, "step": "6.2.3", "name": "弱場長期・多系統統合（集約）"},
        "inputs": {
            "weak_field_test_matrix_json": _relpath(matrix_path),
            "weak_field_systematics_templates_json": _relpath(templates_path),
            "frozen_parameters": frozen,
        },
        "rerun": {"enabled": bool(rerun), "timeout_s": float(timeout_s), "status": rerun_status or None, "log": _relpath(run_log)},
        "results": results,
        "outputs": {
            "weak_field_longterm_consistency_json": "output/summary/weak_field_longterm_consistency.json",
            "weak_field_longterm_consistency_png": "output/summary/weak_field_longterm_consistency.png",
        },
    }
    meta: Dict[str, Any] = {"matrix": matrix, "templates": templates, "frozen": frozen}
    return payload, meta


def _plot(payload: Dict[str, Any], *, out_png: Path) -> None:
    _set_japanese_font()
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting") from e

    res = payload.get("results") or {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()

    # 1) Cassini
    ax = axes[0]
    cass = res.get("cassini_sce1_doppler") or {}
    if cass.get("status") == "ok":
        windows = cass.get("windows") or {}
        xs: List[str] = []
        ys: List[float] = []
        for key in ["all (available points)", "-10 to +10 days", "-3 to +3 days"]:
            if key not in windows:
                continue
            rmse = (windows.get(key) or {}).get("rmse")
            if isinstance(rmse, (int, float)):
                xs.append(key.replace("all (available points)", "all").replace("-10 to +10 days", "±10d").replace("-3 to +3 days", "±3d"))
                ys.append(float(rmse) * 1e11)
        if xs:
            ax.bar(xs, ys)
            ax.set_ylabel("RMSE [×1e-11]")
            ax.set_title("Cassini SCE1：y(t) RMSE（窓別）")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.02, 0.5, "no RMSE rows", transform=ax.transAxes)
    else:
        ax.text(0.02, 0.5, "missing", transform=ax.transAxes)
        ax.set_title("Cassini SCE1")

    # 2) GPS
    ax = axes[1]
    gps = res.get("gps_satellite_clock") or {}
    if gps.get("status") == "ok":
        m = gps.get("metrics") or {}
        brdc_med = m.get("brdc_rms_ns_median")
        pmod_med = m.get("pmodel_rms_ns_median")
        brdc_max = m.get("brdc_rms_ns_max")
        pmod_max = m.get("pmodel_rms_ns_max")
        labels = ["median", "max"]
        x = [0, 1]
        width = 0.35
        b = [float(brdc_med), float(brdc_max)] if isinstance(brdc_med, (int, float)) and isinstance(brdc_max, (int, float)) else None
        p = [float(pmod_med), float(pmod_max)] if isinstance(pmod_med, (int, float)) and isinstance(pmod_max, (int, float)) else None
        if b and p:
            ax.bar([xi - width / 2 for xi in x], b, width, label="BRDC")
            ax.bar([xi + width / 2 for xi in x], p, width, label="P-model")
            ax.set_xticks(x, labels)
            ax.set_ylabel("RMS [ns]")
            ax.set_title("GPS：IGS clock 残差RMS")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
        else:
            ax.text(0.02, 0.5, "missing metrics", transform=ax.transAxes)
    else:
        ax.text(0.02, 0.5, "missing", transform=ax.transAxes)
        ax.set_title("GPS")

    # 3) LLR
    ax = axes[2]
    llr = res.get("llr_batch") or {}
    if llr.get("status") == "ok":
        m = llr.get("median_rms_ns") or {}
        keys = [
            ("tropo+tide", "station_reflector_tropo_tide"),
            ("no-shapiro", "station_reflector_tropo_no_shapiro"),
            ("earth-shapiro", "station_reflector_tropo_earth_shapiro"),
        ]
        xs: List[str] = []
        ys: List[float] = []
        for label, k in keys:
            v = m.get(k)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                xs.append(label)
                ys.append(float(v))
        if xs:
            ax.bar(xs, ys)
            ax.set_yscale("log")
            ax.set_ylabel("median RMS [ns] (log)")
            ax.set_title("LLR：中央値RMS（モデル差）")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.02, 0.5, "no RMS rows", transform=ax.transAxes)
    else:
        ax.text(0.02, 0.5, "missing", transform=ax.transAxes)
        ax.set_title("LLR")

    # 4) Mercury
    ax = axes[3]
    mer = res.get("mercury_perihelion_precession") or {}
    if mer.get("status") == "ok":
        m = mer.get("metrics") or {}
        ref = m.get("reference_arcsec_century")
        pmod = m.get("pmodel_arcsec_century")
        ein = m.get("einstein_arcsec_century")
        xs: List[str] = []
        ys: List[float] = []
        for label, v in [("ref", ref), ("Einstein", ein), ("P-model", pmod)]:
            if isinstance(v, (int, float)):
                xs.append(label)
                ys.append(float(v))
        if xs:
            ax.bar(xs, ys)
            ax.set_ylabel("arcsec/century")
            ax.set_title("Mercury：近日点移動（比較）")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.02, 0.5, "missing values", transform=ax.transAxes)
    else:
        ax.text(0.02, 0.5, "missing", transform=ax.transAxes)
        ax.set_title("Mercury")

    # 5) Viking
    ax = axes[4]
    vik = res.get("viking_shapiro_peak") or {}
    if vik.get("status") == "ok":
        series = vik.get("series") or {}
        vmax = series.get("max_delay_us")
        if isinstance(vmax, (int, float)):
            ax.bar(["max delay"], [float(vmax)])
            ax.set_ylabel("μs")
            ax.set_title("Viking：Shapiro遅延（理論最大）")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.text(0.02, 0.5, "missing max", transform=ax.transAxes)
    else:
        ax.text(0.02, 0.5, "missing", transform=ax.transAxes)
        ax.set_title("Viking")

    # 6) Notes / status
    ax = axes[5]
    ax.axis("off")
    gen = payload.get("generated_utc")
    ax.text(0.0, 0.9, f"generated_utc: {gen}", fontsize=10)
    rerun = payload.get("rerun") or {}
    ax.text(0.0, 0.8, f"rerun: {bool(rerun.get('enabled'))}", fontsize=10)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate weak-field test consistency metrics (Phase 6 / Step 6.2.3).")
    ap.add_argument(
        "--matrix",
        type=str,
        default=str(_ROOT / "output" / "summary" / "weak_field_test_matrix.json"),
        help="Input matrix JSON (default: output/summary/weak_field_test_matrix.json).",
    )
    ap.add_argument(
        "--templates",
        type=str,
        default=str(_ROOT / "output" / "summary" / "weak_field_systematics_templates.json"),
        help="Input systematics templates JSON (default: output/summary/weak_field_systematics_templates.json).",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(_ROOT / "output" / "summary" / "weak_field_longterm_consistency.json"),
        help="Output JSON path (default: output/summary/weak_field_longterm_consistency.json).",
    )
    ap.add_argument(
        "--out-png",
        type=str,
        default=str(_ROOT / "output" / "summary" / "weak_field_longterm_consistency.png"),
        help="Output PNG path (default: output/summary/weak_field_longterm_consistency.png).",
    )
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run baseline commands before collecting (offline-first; uses caches when possible).",
    )
    ap.add_argument(
        "--timeout-seconds",
        type=float,
        default=1800.0,
        help="Timeout seconds for each rerun command (default: 1800).",
    )
    ap.add_argument(
        "--tests",
        type=str,
        default="",
        help="Comma-separated test ids to include (default: all supported tests).",
    )
    args = ap.parse_args(argv)

    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = (_ROOT / matrix_path).resolve()
    templates_path = Path(args.templates)
    if not templates_path.is_absolute():
        templates_path = (_ROOT / templates_path).resolve()
    if not matrix_path.exists():
        print(f"[err] missing matrix: {matrix_path}")
        return 2
    if not templates_path.exists():
        print(f"[err] missing templates: {templates_path}")
        return 2

    tests_filter = None
    if str(args.tests).strip():
        tests_filter = [s.strip() for s in str(args.tests).split(",") if s.strip()]

    payload, _meta = collect(
        matrix_path=matrix_path,
        templates_path=templates_path,
        rerun=bool(args.rerun),
        timeout_s=float(args.timeout_seconds),
        tests_filter=tests_filter,
    )

    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = (_ROOT / out_json).resolve()
    out_png = Path(args.out_png)
    if not out_png.is_absolute():
        out_png = (_ROOT / out_png).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    _plot(payload, out_png=out_png)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {_relpath(out_json)}")
    print(f"[ok] wrote: {_relpath(out_png)}")

    worklog.append_event(
        {
            "event_type": "summary_weak_field_longterm_consistency",
            "phase": "6.2.3",
            "inputs": {"weak_field_test_matrix_json": _relpath(matrix_path), "weak_field_systematics_templates_json": _relpath(templates_path)},
            "outputs": {"weak_field_longterm_consistency_json": _relpath(out_json), "weak_field_longterm_consistency_png": _relpath(out_png)},
            "rerun": bool(args.rerun),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
