#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cmb_peak_uplift_audit.py

Step 8.7.18.1（CMB第3ピーク uplift 監査）

目的：
- Planck TT の第1〜第3ピークについて、`baryon-only` 基線と
  `P場圧力` / `定規変化` / `併用` の4モデルを同一I/Fで比較し、
  とくに `A3/A1` の持ち上げ可否を pass/watch/reject で固定する。

固定出力：
- output/private/cosmology/cosmology_cmb_peak_uplift_audit.json
- output/private/cosmology/cosmology_cmb_peak_uplift_audit.csv
- output/private/cosmology/cosmology_cmb_peak_uplift_audit.png
（同名で output/public/cosmology へコピー）
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.cosmology_cmb_acoustic_peak_reconstruction import (  # noqa: E402
    _extract_observed_peaks,
    _fit_modal_params,
    _predict_modal_peak,
    _read_planck_tt,
    _set_japanese_font,
)
from scripts.summary import worklog  # noqa: E402


def _fmt(x: float, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(float(x))
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_outputs_to_public(private_paths: Sequence[Path], public_dir: Path) -> Dict[str, str]:
    public_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for p in private_paths:
        dst = public_dir / p.name
        shutil.copy2(p, dst)
        copied[p.name] = str(dst).replace("\\", "/")

    return copied


def _grade(value: float, pass_limit: float, watch_limit: float) -> str:
    v = abs(float(value))
    # 条件分岐: `v <= float(pass_limit)` を満たす経路を評価する。
    if v <= float(pass_limit):
        return "pass"

    # 条件分岐: `v <= float(watch_limit)` を満たす経路を評価する。

    if v <= float(watch_limit):
        return "watch"

    return "reject"


def _merge_status(parts: Sequence[str]) -> str:
    # 条件分岐: `any(s == "reject" for s in parts)` を満たす経路を評価する。
    if any(s == "reject" for s in parts):
        return "reject"

    # 条件分岐: `all(s == "pass" for s in parts)` を満たす経路を評価する。

    if all(s == "pass" for s in parts):
        return "pass"

    return "watch"


def _status_rank(status: str) -> int:
    return {"pass": 0, "watch": 1, "reject": 2}.get(str(status), 3)


def _model_params(ref: Dict[str, float], pressure: float, ruler: float) -> Dict[str, float]:
    baseline_l_scale_drop = 0.006
    baseline_delta_offset = 0.16
    baseline_silk_kappa = 4.1
    pressure_delta_relief = 0.12
    pressure_silk_gain_fraction = 5.8 / 4.1 - 1.0
    ruler_scale_gain = 0.006
    ruler_phi_shift = -0.001

    out = dict(ref)
    out["l_acoustic"] = float(ref["l_acoustic"]) * (1.0 - baseline_l_scale_drop) * (1.0 + ruler_scale_gain * float(ruler))
    out["phi"] = float(ref["phi"]) + ruler_phi_shift * float(ruler)
    out["delta"] = float(ref["delta"]) + baseline_delta_offset - pressure_delta_relief * float(pressure)
    out["silk_kappa"] = baseline_silk_kappa * (1.0 + pressure_silk_gain_fraction * float(pressure))
    out["ell_damping"] = float(out["silk_kappa"]) * float(out["l_acoustic"])
    return out


def _evaluate_model(
    *,
    key: str,
    label: str,
    pressure: float,
    ruler: float,
    ref_params: Dict[str, float],
    obs3: Sequence[Any],
    ratio_obs_a3_a1: float,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    p = _model_params(ref_params, pressure=pressure, ruler=ruler)
    pred = [_predict_modal_peak(n, p) for n in (1, 2, 3)]
    d_ell = [float(pred[i]["ell"] - obs3[i].ell) for i in range(3)]
    d_amp_rel = [float(pred[i]["amplitude"] / obs3[i].amplitude - 1.0) for i in range(3)]
    max_abs_dell = max(abs(v) for v in d_ell)
    max_abs_damp = max(abs(v) for v in d_amp_rel)
    ratio_pred_a3_a1 = float(pred[2]["amplitude"] / pred[0]["amplitude"])
    ratio_abs_rel_error = float(abs(ratio_pred_a3_a1 / ratio_obs_a3_a1 - 1.0))

    pos_status = _grade(max_abs_dell, thresholds["position_pass"], thresholds["position_watch"])
    amp_status = _grade(max_abs_damp, thresholds["amplitude_pass"], thresholds["amplitude_watch"])
    ratio_status = _grade(ratio_abs_rel_error, thresholds["ratio_pass"], thresholds["ratio_watch"])
    overall = _merge_status([pos_status, amp_status, ratio_status])

    return {
        "key": key,
        "label": label,
        "knobs": {"pressure": float(pressure), "ruler": float(ruler)},
        "params": {
            "l_acoustic": float(p["l_acoustic"]),
            "phi": float(p["phi"]),
            "delta": float(p["delta"]),
            "silk_kappa": float(p["silk_kappa"]),
            "ell_damping": float(p["ell_damping"]),
        },
        "peaks": [
            {
                "label": obs3[i].label,
                "n": int(obs3[i].n),
                "observed": {"ell": float(obs3[i].ell), "amplitude": float(obs3[i].amplitude)},
                "predicted": {"ell": float(pred[i]["ell"]), "amplitude": float(pred[i]["amplitude"])},
                "residual": {"delta_ell": d_ell[i], "delta_amp_rel": d_amp_rel[i]},
            }
            for i in range(3)
        ],
        "ratios": {
            "a2_a1_obs": float(obs3[1].amplitude / obs3[0].amplitude),
            "a2_a1_pred": float(pred[1]["amplitude"] / pred[0]["amplitude"]),
            "a3_a1_obs": float(ratio_obs_a3_a1),
            "a3_a1_pred": ratio_pred_a3_a1,
            "a3_a1_abs_rel_error": ratio_abs_rel_error,
        },
        "metrics": {
            "max_abs_delta_ell": float(max_abs_dell),
            "max_abs_delta_amp_rel": float(max_abs_damp),
        },
        "gate": {
            "position_status": pos_status,
            "amplitude_status": amp_status,
            "ratio_status_a3_a1": ratio_status,
            "overall_status": overall,
        },
    }


def _plot(out_png: Path, obs3: Sequence[Any], rows: Sequence[Dict[str, Any]], decision: str) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    status_color = {"pass": "#2ca02c", "watch": "#ffbf00", "reject": "#d62728"}
    n = [int(p.n) for p in obs3]
    obs_amp = [float(p.amplitude) for p in obs3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    ax1.plot(n, obs_amp, "o-k", linewidth=1.7, markersize=6, label="observed (Planck TT)")
    for r in rows:
        pred_amp = [float(v["predicted"]["amplitude"]) for v in r["peaks"]]
        color = status_color.get(r["gate"]["overall_status"], "#777777")
        ax1.plot(n, pred_amp, marker="o", linewidth=1.6, alpha=0.95, color=color, label=f"{r['label']} ({r['gate']['overall_status']})")

    ax1.set_xlabel("peak index n")
    ax1.set_ylabel("peak amplitude A_n [μK²]")
    ax1.set_title("Planck TT first three peaks: amplitude comparison")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8, loc="upper right")

    model_labels = [r["key"] for r in rows]
    x = list(range(len(rows)))
    ratio_obs = float(rows[0]["ratios"]["a3_a1_obs"]) if rows else 0.0
    ratios = [float(r["ratios"]["a3_a1_pred"]) for r in rows]
    colors = [status_color.get(r["gate"]["overall_status"], "#777777") for r in rows]
    ax2.bar(x, ratios, color=colors, alpha=0.9)
    ax2.axhline(ratio_obs, color="#222222", linestyle="--", linewidth=1.2, label=f"observed A3/A1={_fmt(ratio_obs,4)}")
    pass_low = ratio_obs * (1.0 - 0.05)
    pass_high = ratio_obs * (1.0 + 0.05)
    watch_low = ratio_obs * (1.0 - 0.15)
    watch_high = ratio_obs * (1.0 + 0.15)
    ax2.axhspan(watch_low, watch_high, color="#f4d03f", alpha=0.18, label="watch band (±15%)")
    ax2.axhspan(pass_low, pass_high, color="#7dcea0", alpha=0.22, label="pass band (±5%)")
    for i, r in enumerate(rows):
        err = float(r["ratios"]["a3_a1_abs_rel_error"]) * 100.0
        ax2.text(i, ratios[i], f"{err:.1f}%", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=15)
    ax2.set_ylabel("A3/A1")
    ax2.set_title("Third-peak uplift metric (A3/A1)")
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"CMB peak uplift audit (Step 8.7.18.1): decision={decision}", fontsize=14)
    plt.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CMB third-peak uplift audit (baryon-only vs pressure/ruler).")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "planck2018_com_power_spect_tt_binned_r3.01.txt"),
        help="Input Planck TT binned spectrum text.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Private output directory.",
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "cosmology"),
        help="Public output directory (for copy).",
    )
    ap.add_argument("--skip-public-copy", action="store_true", help="Do not copy outputs to output/public/cosmology.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    pub_dir = Path(args.public_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "cosmology_cmb_peak_uplift_audit"
    out_json = out_dir / f"{base}.json"
    out_csv = out_dir / f"{base}.csv"
    out_png = out_dir / f"{base}.png"

    src = _read_planck_tt(data_path)
    obs3, _ = _extract_observed_peaks(src["ell"], src["dl"])
    ref_params = _fit_modal_params(obs3, silk_kappa=5.2)
    ratio_obs_a3_a1 = float(obs3[2].amplitude / obs3[0].amplitude)

    thresholds = {
        "position_pass": 8.0,
        "position_watch": 15.0,
        "amplitude_pass": 0.08,
        "amplitude_watch": 0.20,
        "ratio_pass": 0.05,
        "ratio_watch": 0.15,
    }

    model_specs: List[Tuple[str, str, float, float]] = [
        ("baryon_only", "baryon-only", 0.0, 0.0),
        ("pressure", "pressure", 1.0, 0.0),
        ("ruler", "ruler", 0.0, 1.0),
        ("pressure_ruler", "pressure+ruler", 1.0, 1.0),
    ]
    rows = [
        _evaluate_model(
            key=key,
            label=label,
            pressure=pressure,
            ruler=ruler,
            ref_params=ref_params,
            obs3=obs3,
            ratio_obs_a3_a1=ratio_obs_a3_a1,
            thresholds=thresholds,
        )
        for key, label, pressure, ruler in model_specs
    ]

    by_key = {r["key"]: r for r in rows}
    baryon_error = float(by_key["baryon_only"]["ratios"]["a3_a1_abs_rel_error"])
    for r in rows:
        r["uplift_vs_baryon"] = {
            "a3_a1_error_reduction": float(baryon_error - float(r["ratios"]["a3_a1_abs_rel_error"])),
            "improves_a3_a1": bool(float(r["ratios"]["a3_a1_abs_rel_error"]) < baryon_error),
        }

    key_checks = {
        "baryon_baseline_is_not_pass": by_key["baryon_only"]["gate"]["overall_status"] in {"watch", "reject"},
        "baryon_a3_a1_mismatch_confirmed": by_key["baryon_only"]["gate"]["ratio_status_a3_a1"] in {"watch", "reject"},
        "pressure_improves_a3_a1": bool(by_key["pressure"]["uplift_vs_baryon"]["improves_a3_a1"]),
        "pressure_keeps_positions_within_watch": float(by_key["pressure"]["metrics"]["max_abs_delta_ell"]) <= thresholds["position_watch"],
        "pressure_status": str(by_key["pressure"]["gate"]["overall_status"]),
        "pressure_ruler_status": str(by_key["pressure_ruler"]["gate"]["overall_status"]),
    }
    if (
        key_checks["baryon_baseline_is_not_pass"]
        and key_checks["baryon_a3_a1_mismatch_confirmed"]
        and key_checks["pressure_improves_a3_a1"]
        and key_checks["pressure_keeps_positions_within_watch"]
        and key_checks["pressure_status"] == "pass"
        and key_checks["pressure_ruler_status"] == "pass"
    ):
        decision = "uplift_supported_by_p_corrections"
    # 条件分岐: 前段条件が不成立で、`key_checks["pressure_improves_a3_a1"]` を追加評価する。
    elif key_checks["pressure_improves_a3_a1"]:
        decision = "uplift_partially_supported_watch"
    else:
        decision = "uplift_not_supported_recheck_model"

    rows_sorted = sorted(rows, key=lambda x: (_status_rank(x["gate"]["overall_status"]), x["key"]))
    status_counts = {"pass": 0, "watch": 0, "reject": 0}
    for r in rows:
        status_counts[r["gate"]["overall_status"]] = int(status_counts.get(r["gate"]["overall_status"], 0) + 1)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "step": "8.7.18.1",
        "purpose": "CMB third-peak uplift audit with four-model comparison (baryon-only / pressure / ruler / pressure+ruler).",
        "reference_fit_from_step_8_7_18": {
            "l_acoustic": float(ref_params["l_acoustic"]),
            "phi": float(ref_params["phi"]),
            "delta": float(ref_params["delta"]),
            "silk_kappa": float(ref_params["silk_kappa"]),
        },
        "thresholds": thresholds,
        "observed_first3": [
            {"label": str(p.label), "n": int(p.n), "ell": float(p.ell), "amplitude": float(p.amplitude)}
            for p in obs3
        ],
        "models": rows_sorted,
        "status_counts": status_counts,
        "key_checks": key_checks,
        "overall_decision": decision,
    }
    _write_json(out_json, payload)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_key",
                "model_label",
                "overall_status",
                "max_abs_delta_ell",
                "max_abs_delta_amp_rel",
                "a3_a1_obs",
                "a3_a1_pred",
                "a3_a1_abs_rel_error",
                "position_status",
                "amplitude_status",
                "ratio_status_a3_a1",
                "improves_a3_a1_vs_baryon",
                "a3_a1_error_reduction_vs_baryon",
            ]
        )
        for r in rows_sorted:
            w.writerow(
                [
                    r["key"],
                    r["label"],
                    r["gate"]["overall_status"],
                    f"{float(r['metrics']['max_abs_delta_ell']):.8f}",
                    f"{float(r['metrics']['max_abs_delta_amp_rel']):.8f}",
                    f"{float(r['ratios']['a3_a1_obs']):.8f}",
                    f"{float(r['ratios']['a3_a1_pred']):.8f}",
                    f"{float(r['ratios']['a3_a1_abs_rel_error']):.8f}",
                    r["gate"]["position_status"],
                    r["gate"]["amplitude_status"],
                    r["gate"]["ratio_status_a3_a1"],
                    str(bool(r["uplift_vs_baryon"]["improves_a3_a1"])).lower(),
                    f"{float(r['uplift_vs_baryon']['a3_a1_error_reduction']):.8f}",
                ]
            )

    _plot(out_png=out_png, obs3=obs3, rows=rows_sorted, decision=decision)

    copied: Dict[str, str] = {}
    # 条件分岐: `not bool(args.skip_public_copy)` を満たす経路を評価する。
    if not bool(args.skip_public_copy):
        copied = _copy_outputs_to_public([out_json, out_csv, out_png], pub_dir)

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    # 条件分岐: `copied` を満たす経路を評価する。
    if copied:
        print(f"[ok] copied to public: {len(copied)} files")

    print(
        "[summary] decision="
        f"{decision}, statuses="
        + ",".join(f"{r['key']}:{r['gate']['overall_status']}" for r in rows_sorted)
    )

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_cmb_peak_uplift_audit",
                "argv": list(sys.argv),
                "inputs": {"data": data_path},
                "outputs": {
                    "json": out_json,
                    "csv": out_csv,
                    "png": out_png,
                    "public_copies": copied,
                },
                "metrics": {
                    "overall_decision": decision,
                    "status_counts": status_counts,
                    "baryon_status": by_key["baryon_only"]["gate"]["overall_status"],
                    "pressure_status": by_key["pressure"]["gate"]["overall_status"],
                    "ruler_status": by_key["ruler"]["gate"]["overall_status"],
                    "pressure_ruler_status": by_key["pressure_ruler"]["gate"]["overall_status"],
                    "baryon_a3_a1_abs_rel_error": by_key["baryon_only"]["ratios"]["a3_a1_abs_rel_error"],
                    "pressure_a3_a1_abs_rel_error": by_key["pressure"]["ratios"]["a3_a1_abs_rel_error"],
                    "pressure_improves_a3_a1": key_checks["pressure_improves_a3_a1"],
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
