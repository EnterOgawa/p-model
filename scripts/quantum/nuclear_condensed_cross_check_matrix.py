#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nuclear_condensed_cross_check_matrix.py

Step 8.7.15:
核（B.E./分離エネルギー/半径）と物性・熱 holdout を同一スコア軸へ射影し、
横断 cross-check matrix を固定出力する。

出力:
  - output/public/quantum/nuclear_condensed_cross_check_matrix.json
  - output/public/quantum/nuclear_condensed_cross_check_matrix.csv
  - output/public/quantum/nuclear_condensed_cross_check_matrix.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isfinite(v):
            return v
    return None


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0.0:
        return None
    return float(num) / float(den)


def _score_status(score: Optional[float]) -> str:
    if score is None:
        return "watch"
    if score <= 1.0:
        return "pass"
    if score <= 1.5:
        return "watch"
    return "reject"


def _pick_model(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        mid = str(row.get("model_id") or "")
        if "nu_saturation" in mid:
            return row
    for row in rows:
        if not isinstance(row, dict):
            continue
        passes = row.get("passes") if isinstance(row.get("passes"), dict) else {}
        if bool(passes.get("median_within_3sigma")) and bool(passes.get("a_trend_within_3sigma")):
            return row
    for row in rows:
        if isinstance(row, dict):
            return row
    return None


def _extract_nuclear_be(minphys: Dict[str, Any]) -> Dict[str, Any]:
    models = minphys.get("models") if isinstance(minphys.get("models"), list) else []
    model = _pick_model(models)
    thresholds = minphys.get("thresholds") if isinstance(minphys.get("thresholds"), dict) else {}
    z_thr = _as_float(thresholds.get("z_median_abs_max")) or 3.0
    z_delta_thr = _as_float(thresholds.get("z_delta_median_abs_max")) or 3.0

    z_median = _as_float(model.get("z_median")) if isinstance(model, dict) else None
    z_delta = _as_float(model.get("z_delta_median")) if isinstance(model, dict) else None
    z_abs_max = max(abs(z_median or 0.0), abs(z_delta or 0.0)) if (z_median is not None or z_delta is not None) else None
    score = _safe_div(z_abs_max, max(z_thr, z_delta_thr))

    return {
        "axis_id": "nuclear_be_core",
        "title": "Nuclear B.E. core (ν saturation)",
        "operator": "<=",
        "value": z_abs_max,
        "threshold": max(z_thr, z_delta_thr),
        "score": score,
        "status": _score_status(score),
        "details": {
            "model_id": model.get("model_id") if isinstance(model, dict) else None,
            "z_median": z_median,
            "z_delta_median": z_delta,
            "z_median_threshold": z_thr,
            "z_delta_threshold": z_delta_thr,
        },
    }


def _extract_nuclear_sep_radius(nuclear_pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    independent = nuclear_pack.get("independent_cross_checks") if isinstance(nuclear_pack.get("independent_cross_checks"), dict) else {}
    gate_status = independent.get("gate_status") if isinstance(independent.get("gate_status"), dict) else {}

    sep = independent.get("separation_energies") if isinstance(independent.get("separation_energies"), dict) else {}
    sn_rms_total = _as_float(sep.get("sn_rms_total_mev"))
    s2n_rms_total = _as_float(sep.get("s2n_rms_total_mev"))
    gap_sn = _as_float(sep.get("gap_sn_rms_median_mev"))
    gap_s2n = _as_float(sep.get("gap_s2n_rms_median_mev"))
    ratio_sn = _safe_div(gap_sn, sn_rms_total)
    ratio_s2n = _safe_div(gap_s2n, s2n_rms_total)
    sep_ratio_candidates = [x for x in (ratio_sn, ratio_s2n) if x is not None]
    sep_ratio = max(sep_ratio_candidates) if sep_ratio_candidates else None
    sep_available = bool(gate_status.get("separation_gap_available"))
    sep_score = sep_ratio if sep_available else None
    sep_status = _score_status(sep_score)
    if not sep_available and sep_status == "watch":
        sep_status = "reject"

    radius = independent.get("charge_radius_kink") if isinstance(independent.get("charge_radius_kink"), dict) else {}
    strict_thr = _as_float(radius.get("strict_threshold_sigma")) or 3.0
    pairing = radius.get("pairing_def0_fit") if isinstance(radius.get("pairing_def0_fit"), dict) else {}
    max_sn = _as_float(pairing.get("max_abs_resid_sigma_sn"))
    max_sp = _as_float(pairing.get("max_abs_resid_sigma_sp"))
    max_sigma = max(x for x in (max_sn, max_sp) if x is not None) if (max_sn is not None or max_sp is not None) else None
    radius_score = _safe_div(max_sigma, strict_thr)

    sep_axis = {
        "axis_id": "nuclear_separation",
        "title": "Nuclear separation-gap consistency",
        "operator": "<=",
        "value": sep_ratio,
        "threshold": 1.0,
        "score": sep_score,
        "status": sep_status,
        "details": {
            "separation_gap_available": sep_available,
            "ratio_sn_gap_over_total_rms": ratio_sn,
            "ratio_s2n_gap_over_total_rms": ratio_s2n,
            "gap_sn_rms_median_mev": gap_sn,
            "gap_s2n_rms_median_mev": gap_s2n,
            "sn_rms_total_mev": sn_rms_total,
            "s2n_rms_total_mev": s2n_rms_total,
        },
    }
    radius_axis = {
        "axis_id": "nuclear_radius_kink",
        "title": "Nuclear charge-radius kink strict",
        "operator": "<=",
        "value": max_sigma,
        "threshold": strict_thr,
        "score": radius_score,
        "status": _score_status(radius_score),
        "details": {
            "radius_strict_pass_A100_pairing": bool(gate_status.get("radius_strict_pass_A100_pairing")),
            "max_abs_resid_sigma_sn": max_sn,
            "max_abs_resid_sigma_sp": max_sp,
        },
    }
    return {"separation": sep_axis, "radius": radius_axis}


def _extract_condensed(condensed_holdout: Dict[str, Any]) -> Dict[str, Any]:
    summary = condensed_holdout.get("summary") if isinstance(condensed_holdout.get("summary"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    included = [row for row in datasets if isinstance(row, dict) and bool(row.get("kpi_include", True))]

    ok_n = 0
    reject_n = 0
    inconclusive_n = 0
    for row in included:
        gates = row.get("audit_gates") if isinstance(row.get("audit_gates"), dict) else {}
        fals = gates.get("falsification") if isinstance(gates.get("falsification"), dict) else {}
        status = str(fals.get("status") or "")
        if status == "ok":
            ok_n += 1
        elif status == "reject":
            reject_n += 1
        else:
            inconclusive_n += 1

    total = len(included)
    reject_ratio = (float(reject_n) / float(total)) if total > 0 else None
    inconclusive_ratio = (float(inconclusive_n) / float(total)) if total > 0 else None
    reject_thr = 0.10
    reject_score = _safe_div(reject_ratio, reject_thr)

    if inconclusive_n == 0:
        score = reject_score
    else:
        score = None

    status = _score_status(score)
    if inconclusive_n > 0 and status == "watch":
        status = "reject"

    return {
        "axis_id": "condensed_holdout",
        "title": "Condensed/thermal holdout",
        "operator": "<=",
        "value": reject_ratio,
        "threshold": reject_thr,
        "score": score,
        "status": status,
        "details": {
            "included_datasets_n": total,
            "ok_n": ok_n,
            "reject_n": reject_n,
            "inconclusive_n": inconclusive_n,
            "inconclusive_ratio": inconclusive_ratio,
            "requirement": "inconclusive_n must be 0 for pass/watch gating",
        },
    }


def _extract_nuclear_holdout_context(nuclear_holdout: Dict[str, Any]) -> Dict[str, Optional[float]]:
    groups = nuclear_holdout.get("groups") if isinstance(nuclear_holdout.get("groups"), list) else []
    context = {
        "all_outlier_frac": None,
        "near_magic_outlier_frac": None,
        "deformed_outlier_frac": None,
    }
    for row in groups:
        if not isinstance(row, dict):
            continue
        gid = str(row.get("group_id") or "")
        frac = _as_float(row.get("outlier_abs_z_gt3_frac"))
        if gid == "all":
            context["all_outlier_frac"] = frac
        elif gid == "near_magic_width2":
            context["near_magic_outlier_frac"] = frac
        elif gid == "deformed_abs_beta2_ge_0p20_nonmagic":
            context["deformed_outlier_frac"] = frac
    return context


def _build_matrix(axes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for axis_i in axes:
        id_i = str(axis_i.get("axis_id") or "")
        score_i = _as_float(axis_i.get("score"))
        for axis_j in axes:
            id_j = str(axis_j.get("axis_id") or "")
            score_j = _as_float(axis_j.get("score"))
            if score_i is None or score_j is None:
                pair_score = None
                pair_status = "watch"
            else:
                pair_score = max(score_i, score_j)
                pair_status = _score_status(pair_score)
            rows.append(
                {
                    "axis_i": id_i,
                    "axis_j": id_j,
                    "score_i": score_i,
                    "score_j": score_j,
                    "pair_score_max": pair_score,
                    "pair_status": pair_status,
                }
            )
    return rows


def _write_csv(out_csv: Path, axes: List[Dict[str, Any]], matrix_rows: List[Dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "axis_id", "title", "status", "value", "threshold", "score", "details_json"])
        for axis in axes:
            writer.writerow(
                [
                    "axes",
                    axis.get("axis_id"),
                    axis.get("title"),
                    axis.get("status"),
                    axis.get("value"),
                    axis.get("threshold"),
                    axis.get("score"),
                    json.dumps(axis.get("details"), ensure_ascii=False, separators=(",", ":")),
                ]
            )
        writer.writerow([])
        writer.writerow(["section", "axis_i", "axis_j", "pair_score_max", "pair_status"])
        for row in matrix_rows:
            writer.writerow(["matrix", row.get("axis_i"), row.get("axis_j"), row.get("pair_score_max"), row.get("pair_status")])


def _plot(
    *,
    out_png: Path,
    axes: List[Dict[str, Any]],
    matrix_rows: List[Dict[str, Any]],
    context: Dict[str, Optional[float]],
    condensed_axis: Dict[str, Any],
) -> None:
    labels = [str(a.get("title") or a.get("axis_id") or "") for a in axes]
    scores = [(_as_float(a.get("score")) if _as_float(a.get("score")) is not None else np.nan) for a in axes]
    status_colors = []
    for axis in axes:
        status = str(axis.get("status") or "watch")
        if status == "pass":
            status_colors.append("#2f9e44")
        elif status == "watch":
            status_colors.append("#f2c94c")
        else:
            status_colors.append("#e03131")

    n = len(axes)
    mat = np.full((n, n), np.nan, dtype=float)
    id_to_idx = {str(a.get("axis_id")): i for i, a in enumerate(axes)}
    for row in matrix_rows:
        i = id_to_idx.get(str(row.get("axis_i")))
        j = id_to_idx.get(str(row.get("axis_j")))
        if i is None or j is None:
            continue
        v = _as_float(row.get("pair_score_max"))
        if v is not None:
            mat[i, j] = float(v)

    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.2), dpi=180)
    ax00, ax01, ax10, ax11 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    y = np.arange(len(labels))
    ax00.barh(y, scores, color=status_colors)
    ax00.axvline(1.0, color="#444444", linestyle="--", linewidth=1.0)
    ax00.set_yticks(y, labels)
    ax00.set_xlim(0.0, max(1.2, float(np.nanmax(scores)) * 1.15 if np.isfinite(np.nanmax(scores)) else 1.2))
    ax00.set_xlabel("normalized score (<=1 pass)")
    ax00.set_title("Axis scores")
    ax00.grid(axis="x", linestyle=":", alpha=0.35)

    show = np.clip(mat, 0.0, 2.0)
    im = ax01.imshow(show, cmap="RdYlGn_r", vmin=0.0, vmax=2.0)
    ax01.set_xticks(np.arange(n), [str(a.get("axis_id")) for a in axes], rotation=20, ha="right")
    ax01.set_yticks(np.arange(n), [str(a.get("axis_id")) for a in axes])
    ax01.set_title("Cross-check matrix (pair max score)")
    for i in range(n):
        for j in range(n):
            if not np.isfinite(mat[i, j]):
                txt = "n/a"
            else:
                txt = f"{mat[i, j]:.2f}"
            ax01.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax01, fraction=0.046, pad=0.04)

    ctx_labels = ["nuclear_all", "nuclear_near_magic", "nuclear_deformed", "condensed_reject_ratio"]
    ctx_values = [
        context.get("all_outlier_frac"),
        context.get("near_magic_outlier_frac"),
        context.get("deformed_outlier_frac"),
        _as_float(condensed_axis.get("value")),
    ]
    ax10.bar(ctx_labels, [float(v) if v is not None else np.nan for v in ctx_values], color=["#4c78a8", "#f58518", "#54a24b", "#9c755f"])
    ax10.axhline(0.10, color="#444444", linestyle="--", linewidth=1.0)
    ax10.set_ylim(0.0, max(0.12, max((v for v in ctx_values if v is not None), default=0.0) * 1.15))
    ax10.set_ylabel("fraction")
    ax10.set_title("Context: outlier/reject fractions")
    ax10.tick_params(axis="x", rotation=20)
    ax10.grid(axis="y", linestyle=":", alpha=0.35)

    statuses = [str(axis.get("status") or "watch") for axis in axes]
    pass_n = statuses.count("pass")
    watch_n = statuses.count("watch")
    reject_n = statuses.count("reject")
    ax11.bar(["pass", "watch", "reject"], [pass_n, watch_n, reject_n], color=["#2f9e44", "#f2c94c", "#e03131"])
    ax11.set_title("Axis gate counts")
    ax11.set_ylabel("count")
    ax11.grid(axis="y", linestyle=":", alpha=0.35)

    overall = "pass" if reject_n == 0 and watch_n == 0 else ("watch" if reject_n == 0 else "reject")
    fig.suptitle(f"Nuclear / condensed cross-check matrix audit (overall={overall})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def build_payload(
    *,
    minphys_json: Path,
    nuclear_pack_json: Path,
    nuclear_holdout_json: Path,
    condensed_holdout_json: Path,
) -> Dict[str, Any]:
    minphys = _read_json(minphys_json)
    nuclear_pack = _read_json(nuclear_pack_json)
    nuclear_holdout = _read_json(nuclear_holdout_json)
    condensed_holdout = _read_json(condensed_holdout_json)

    axis_be = _extract_nuclear_be(minphys)
    sep_radius = _extract_nuclear_sep_radius(nuclear_pack)
    axis_sep = sep_radius["separation"]
    axis_radius = sep_radius["radius"]
    axis_condensed = _extract_condensed(condensed_holdout)

    axes = [axis_be, axis_sep, axis_radius, axis_condensed]
    matrix_rows = _build_matrix(axes)

    scores = [_as_float(a.get("score")) for a in axes]
    finite_scores = [s for s in scores if s is not None and math.isfinite(s)]
    overall_score = max(finite_scores) if finite_scores else None
    axis_statuses = [str(a.get("status") or "watch") for a in axes]
    if any(s == "reject" for s in axis_statuses):
        overall_status = "reject"
    elif any(s == "watch" for s in axis_statuses):
        overall_status = "watch"
    else:
        overall_status = "pass"

    context = _extract_nuclear_holdout_context(nuclear_holdout)

    return {
        "generated_utc": _utc_now(),
        "phase": {"phase": 8, "step": "8.7.15", "name": "Nuclear-condensed cross-check matrix"},
        "intent": "Integrate nuclear (B.E./separation/radius) and condensed/thermal holdout gates into one cross-check matrix.",
        "inputs": {
            "nuclear_minimal_additional_physics_metrics_json": _rel(minphys_json),
            "nuclear_falsification_pack_json": _rel(nuclear_pack_json),
            "nuclear_holdout_audit_summary_json": _rel(nuclear_holdout_json),
            "condensed_holdout_audit_summary_json": _rel(condensed_holdout_json),
        },
        "thresholds": {
            "normalized_score_pass_le": 1.0,
            "normalized_score_watch_le": 1.5,
            "condensed_reject_ratio_watch_threshold": 0.10,
            "pair_matrix_definition": "pair_score_max = max(score_i, score_j)",
        },
        "axes": axes,
        "matrix": matrix_rows,
        "context": context,
        "overall": {
            "status": overall_status,
            "axis_n": len(axes),
            "axis_status_counts": {
                "pass": axis_statuses.count("pass"),
                "watch": axis_statuses.count("watch"),
                "reject": axis_statuses.count("reject"),
            },
            "max_score": overall_score,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Step 8.7.15 nuclear/condensed cross-check matrix.")
    parser.add_argument(
        "--nuclear-minphys",
        default=str(ROOT / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_minimal_additional_physics_metrics.json"),
        help="Input nuclear minimal-additional-physics metrics JSON.",
    )
    parser.add_argument(
        "--nuclear-pack",
        default=str(ROOT / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_falsification_pack.json"),
        help="Input nuclear falsification pack JSON.",
    )
    parser.add_argument(
        "--nuclear-holdout",
        default=str(ROOT / "output" / "public" / "quantum" / "nuclear_holdout_audit_summary.json"),
        help="Input nuclear holdout summary JSON.",
    )
    parser.add_argument(
        "--condensed-holdout",
        default=str(ROOT / "output" / "public" / "quantum" / "condensed_holdout_audit_summary.json"),
        help="Input condensed holdout summary JSON.",
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "output" / "public" / "quantum" / "nuclear_condensed_cross_check_matrix.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "output" / "public" / "quantum" / "nuclear_condensed_cross_check_matrix.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        default=str(ROOT / "output" / "public" / "quantum" / "nuclear_condensed_cross_check_matrix.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    in_minphys = Path(args.nuclear_minphys).resolve() if Path(args.nuclear_minphys).is_absolute() else (ROOT / args.nuclear_minphys).resolve()
    in_pack = Path(args.nuclear_pack).resolve() if Path(args.nuclear_pack).is_absolute() else (ROOT / args.nuclear_pack).resolve()
    in_n_hold = Path(args.nuclear_holdout).resolve() if Path(args.nuclear_holdout).is_absolute() else (ROOT / args.nuclear_holdout).resolve()
    in_c_hold = Path(args.condensed_holdout).resolve() if Path(args.condensed_holdout).is_absolute() else (ROOT / args.condensed_holdout).resolve()
    out_json = Path(args.out_json).resolve() if Path(args.out_json).is_absolute() else (ROOT / args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve() if Path(args.out_csv).is_absolute() else (ROOT / args.out_csv).resolve()
    out_png = Path(args.out_png).resolve() if Path(args.out_png).is_absolute() else (ROOT / args.out_png).resolve()

    for p in (in_minphys, in_pack, in_n_hold, in_c_hold):
        if not p.exists():
            raise FileNotFoundError(f"required input not found: {_rel(p)}")

    payload = build_payload(
        minphys_json=in_minphys,
        nuclear_pack_json=in_pack,
        nuclear_holdout_json=in_n_hold,
        condensed_holdout_json=in_c_hold,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, payload.get("axes", []), payload.get("matrix", []))
    _plot(
        out_png=out_png,
        axes=payload.get("axes", []),
        matrix_rows=payload.get("matrix", []),
        context=payload.get("context", {}),
        condensed_axis=(payload.get("axes", [None, None, None, {}])[3] if isinstance(payload.get("axes"), list) and len(payload.get("axes", [])) >= 4 else {}),
    )

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "nuclear_condensed_cross_check_matrix",
                "phase": "8.7.15",
                "inputs": {
                    "nuclear_minphys_json": {"path": _rel(in_minphys), "sha256": _sha256(in_minphys)},
                    "nuclear_falsification_pack_json": {"path": _rel(in_pack), "sha256": _sha256(in_pack)},
                    "nuclear_holdout_summary_json": {"path": _rel(in_n_hold), "sha256": _sha256(in_n_hold)},
                    "condensed_holdout_summary_json": {"path": _rel(in_c_hold), "sha256": _sha256(in_c_hold)},
                },
                "outputs": {
                    "json": _rel(out_json),
                    "csv": _rel(out_csv),
                    "png": _rel(out_png),
                },
                "overall": payload.get("overall"),
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
