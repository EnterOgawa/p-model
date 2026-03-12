#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_xi_cross_integration.py

Step 8.7.35.2（BAO xi_l BOSS+DESI 横断統合）

目的:
- BOSS DR12 post-recon xi_l peakfit（Stage A）と
  DESI DR1 multi-tracer promotion check（Stage B）を同一判定I/Fへ統合する。
- Part IV の判定表（Pass/Watch/Reject）を更新すべきかを、
  統合メトリクスから機械判定できるよう固定する。

注意:
- 本スクリプトは「統合監査」用であり、DESI 側の upstream fit/cov 推定は再実行しない。
- 判定規約は既存運用（z閾値=3, promoted gate）に従う。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。
def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None

    # 条件分岐: `not np.isfinite(v)` を満たす経路を評価する。

    if not np.isfinite(v):
        return None

    return float(v)


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_sigma_from_ci` の入出力契約と処理意図を定義する。

def _sigma_from_ci(ci: Any) -> Optional[float]:
    # 条件分岐: `not (isinstance(ci, list) and len(ci) == 2)` を満たす経路を評価する。
    if not (isinstance(ci, list) and len(ci) == 2):
        return None

    lo = _safe_float(ci[0])
    hi = _safe_float(ci[1])
    # 条件分岐: `lo is None or hi is None` を満たす経路を評価する。
    if lo is None or hi is None:
        return None

    sigma = 0.5 * (float(hi) - float(lo))
    # 条件分岐: `sigma <= 0.0` を満たす経路を評価する。
    if sigma <= 0.0:
        return None

    return float(sigma)


# 関数: `_stage_a_status` の入出力契約と処理意図を定義する。

def _stage_a_status(max_abs_z: Optional[float]) -> str:
    # 条件分岐: `max_abs_z is None` を満たす経路を評価する。
    if max_abs_z is None:
        return "watch"

    z = float(abs(max_abs_z))
    # 条件分岐: `z <= 3.0` を満たす経路を評価する。
    if z <= 3.0:
        return "pass"

    # 条件分岐: `z <= 5.0` を満たす経路を評価する。

    if z <= 5.0:
        return "watch"

    return "reject"


# 関数: `_stage_b_status` の入出力契約と処理意図を定義する。

def _stage_b_status(promoted: bool) -> str:
    return "pass" if bool(promoted) else "watch"


# 関数: `_overall_status` の入出力契約と処理意図を定義する。

def _overall_status(stage_a: str, stage_b: str) -> str:
    # 条件分岐: `stage_a == "reject"` を満たす経路を評価する。
    if stage_a == "reject":
        return "reject"

    # 条件分岐: `stage_a == "pass" and stage_b == "pass"` を満たす経路を評価する。

    if stage_a == "pass" and stage_b == "pass":
        return "pass"

    return "watch"


# 関数: `_load_boss_stage_a` の入出力契約と処理意図を定義する。

def _load_boss_stage_a(path: Path) -> Dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("results") if isinstance(payload.get("results"), list) else []
    items: List[Dict[str, Any]] = []
    abs_z_values: List[float] = []

    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        fit = row.get("fit") if isinstance(row.get("fit"), dict) else {}
        free = fit.get("free") if isinstance(fit.get("free"), dict) else {}
        eps = _safe_float(free.get("eps"))
        sigma = _sigma_from_ci(free.get("eps_ci_1sigma"))
        abs_z = None
        # 条件分岐: `eps is not None and sigma is not None` を満たす経路を評価する。
        if eps is not None and sigma is not None:
            abs_z = float(abs(float(eps) / float(sigma)))
            abs_z_values.append(abs_z)

        items.append(
            {
                "zbin": int(_safe_float(row.get("zbin")) or 0),
                "z_eff": _safe_float(row.get("z_eff")),
                "eps": eps,
                "sigma_eps": sigma,
                "abs_z": abs_z,
            }
        )

    max_abs_z = max(abs_z_values) if abs_z_values else None
    stage_status = _stage_a_status(max_abs_z)
    return {
        "source": str(path).replace("\\", "/"),
        "n_bins": len(items),
        "max_abs_z": max_abs_z,
        "status": stage_status,
        "rows": items,
    }


# 関数: `_load_desi_stage_b` の入出力契約と処理意図を定義する。

def _load_desi_stage_b(path: Path) -> Dict[str, Any]:
    payload = _read_json(path)
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    gate_by_tracer = payload.get("gate_by_tracer") if isinstance(payload.get("gate_by_tracer"), dict) else {}
    promoted = bool(result.get("promoted"))
    passing_n = int(result.get("passing_tracers_n") or 0)
    passing = [str(x) for x in (result.get("passing_tracers") or [])]
    stage_status = _stage_b_status(promoted)
    tracer_rows: List[Dict[str, Any]] = []

    for tracer, info in sorted(gate_by_tracer.items()):
        # 条件分岐: `not isinstance(info, dict)` を満たす経路を評価する。
        if not isinstance(info, dict):
            continue

        ranges = info.get("ranges") if isinstance(info.get("ranges"), list) else []
        z_min_values: List[float] = []
        z_max_values: List[float] = []
        sign_flip_any = False
        for rg in ranges:
            # 条件分岐: `not isinstance(rg, dict)` を満たす経路を評価する。
            if not isinstance(rg, dict):
                continue

            z_min = _safe_float(rg.get("z_min"))
            z_max = _safe_float(rg.get("z_max"))
            # 条件分岐: `z_min is not None` を満たす経路を評価する。
            if z_min is not None:
                z_min_values.append(z_min)

            # 条件分岐: `z_max is not None` を満たす経路を評価する。

            if z_max is not None:
                z_max_values.append(z_max)

            sign_flip_any = sign_flip_any or bool(rg.get("sign_flips"))

        tracer_rows.append(
            {
                "tracer": str(tracer),
                "stable_all_methods": bool(info.get("stable_all_methods")),
                "methods_n": int(info.get("methods_n") or 0),
                "z_min": min(z_min_values) if z_min_values else None,
                "z_max": max(z_max_values) if z_max_values else None,
                "sign_flip_any": sign_flip_any,
            }
        )

    return {
        "source": str(path).replace("\\", "/"),
        "promoted": promoted,
        "passing_tracers_n": passing_n,
        "passing_tracers": passing,
        "status": stage_status,
        "rows": tracer_rows,
    }


# 関数: `_part4_update_recommendation` の入出力契約と処理意図を定義する。

def _part4_update_recommendation(overall_status: str) -> Dict[str, Any]:
    # 8.7.35.2 では「更新可否」の監査を明示する。
    # 条件分岐: `overall_status == "pass"` を満たす経路を評価する。
    if overall_status == "pass":
        return {
            "recommend_update": True,
            "reason": "BOSS Stage A と DESI Stage B が同時に pass のため、判定表の昇格更新を推奨。",
        }

    # 条件分岐: `overall_status == "reject"` を満たす経路を評価する。

    if overall_status == "reject":
        return {
            "recommend_update": True,
            "reason": "Stage A が reject のため、判定表へ棄却反映を推奨。",
        }

    return {
        "recommend_update": False,
        "reason": "統合判定は watch（DESI promotion 未達）であり、判定表の総数更新は不要。",
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(
    *,
    boss: Dict[str, Any],
    desi: Dict[str, Any],
    overall_status: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.8, 9.6))

    rows_b = boss.get("rows") if isinstance(boss.get("rows"), list) else []
    z_labels: List[str] = []
    z_values: List[float] = []
    for row in rows_b:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        zbin = int(_safe_float(row.get("zbin")) or 0)
        abs_z = _safe_float(row.get("abs_z"))
        # 条件分岐: `abs_z is None` を満たす経路を評価する。
        if abs_z is None:
            continue

        z_labels.append(f"zbin{zbin}")
        z_values.append(float(abs_z))

    x = np.arange(len(z_values), dtype=float)
    ax0.bar(x, z_values, color="#4e79a7", width=0.66)
    ax0.axhline(3.0, color="#d62728", linestyle="--", linewidth=1.4, label="|z|=3 threshold")
    ax0.axhline(5.0, color="#9467bd", linestyle=":", linewidth=1.2, label="|z|=5 reject line")
    ax0.set_xticks(x)
    ax0.set_xticklabels(z_labels)
    ax0.set_ylabel("BOSS |eps|/sigma_eps", fontsize=15.8)
    ax0.set_title("Stage A: BOSS xi_l peakfit", fontsize=17.4)
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="upper right", fontsize=14.2)
    ax0.tick_params(labelsize=13.8)

    rows_d = desi.get("rows") if isinstance(desi.get("rows"), list) else []
    z_span_min = math.inf
    z_span_max = -math.inf
    for idx, row in enumerate(rows_d):
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        tracer = str(row.get("tracer") or f"tracer{idx+1}")
        z_min = _safe_float(row.get("z_min"))
        z_max = _safe_float(row.get("z_max"))
        # 条件分岐: `z_min is None or z_max is None` を満たす経路を評価する。
        if z_min is None or z_max is None:
            continue

        color = "#59a14f" if bool(row.get("stable_all_methods")) else "#e15759"
        z_span_min = min(z_span_min, z_min)
        z_span_max = max(z_span_max, z_max)
        ax1.plot([z_min, z_max], [idx, idx], color=color, linewidth=5.2, solid_capstyle="round")
        ax1.scatter([z_min, z_max], [idx, idx], color=color, s=32.0, zorder=3)
        ax1.annotate(
            tracer,
            xy=(z_max, idx),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=14.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.15},
            annotation_clip=False,
        )

    ax1.axvline(0.0, color="#444444", linestyle="--", linewidth=1.1)
    ax1.axvline(3.0, color="#d62728", linestyle=":", linewidth=1.2)
    ax1.axvline(-3.0, color="#d62728", linestyle=":", linewidth=1.2)
    ax1.set_yticks([])
    ax1.set_xlabel("DESI z_score_combined range", fontsize=15.8)
    ax1.set_title("Stage B: DESI promotion gate", fontsize=17.4)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax1.tick_params(labelsize=13.8)
    if z_span_min < math.inf and z_span_max > -math.inf:
        ax1.set_xlim(min(-3.6, z_span_min - 0.35), max(3.6, z_span_max + 1.65))

    summary = (
        f"StageA={boss.get('status')} (max|z|={boss.get('max_abs_z'):.3f})\n"
        f"StageB={desi.get('status')} (promoted={bool(desi.get('promoted'))}, passing={int(desi.get('passing_tracers_n') or 0)})\n"
        f"overall={overall_status}"
    )
    fig.text(
        0.18,
        0.026,
        summary,
        ha="left",
        va="bottom",
        fontsize=14.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.92},
    )

    fig.suptitle("BAO xi_l cross integration (BOSS Stage A + DESI Stage B)", fontsize=17.8)
    fig.tight_layout(rect=(0.0, 0.095, 1.0, 0.95), h_pad=2.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: Iterable[Tuple[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key, value in rows:
            writer.writerow([key, value])


# 関数: `_copy_to_public` の入出力契約と処理意図を定義する。

def _copy_to_public(private_files: Iterable[Path], public_dir: Path) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    for src in private_files:
        shutil.copy2(src, public_dir / src.name)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    parser = argparse.ArgumentParser(description="BAO xi_l cross integration (BOSS + DESI) for Step 8.7.35.2.")
    parser.add_argument(
        "--boss-metrics",
        default=str(ROOT / "output" / "public" / "cosmology" / "cosmology_bao_xi_multipole_peakfit_metrics.json"),
        help="Input BOSS xi_l peakfit metrics JSON.",
    )
    parser.add_argument(
        "--desi-metrics",
        default=str(ROOT / "output" / "public" / "cosmology" / "cosmology_desi_dr1_bao_promotion_check.json"),
        help="Input DESI promotion check JSON.",
    )
    parser.add_argument(
        "--out-private-dir",
        default=str(ROOT / "output" / "private" / "cosmology"),
        help="Output directory (private).",
    )
    parser.add_argument(
        "--out-public-dir",
        default=str(ROOT / "output" / "public" / "cosmology"),
        help="Output directory (public mirror).",
    )
    args = parser.parse_args()

    boss_path = Path(args.boss_metrics).resolve()
    desi_path = Path(args.desi_metrics).resolve()
    # 条件分岐: `not boss_path.exists()` を満たす経路を評価する。
    if not boss_path.exists():
        raise FileNotFoundError(f"boss metrics not found: {boss_path}")

    # 条件分岐: `not desi_path.exists()` を満たす経路を評価する。

    if not desi_path.exists():
        raise FileNotFoundError(f"desi metrics not found: {desi_path}")

    boss = _load_boss_stage_a(boss_path)
    desi = _load_desi_stage_b(desi_path)
    overall = _overall_status(str(boss.get("status")), str(desi.get("status")))
    part4_update = _part4_update_recommendation(overall)

    out_private = Path(args.out_private_dir).resolve()
    out_public = Path(args.out_public_dir).resolve()
    out_png = out_private / "cosmology_bao_xi_cross_integration.png"
    out_pdf = out_private / "cosmology_bao_xi_cross_integration.pdf"
    out_json = out_private / "cosmology_bao_xi_cross_integration_metrics.json"
    out_csv = out_private / "cosmology_bao_xi_cross_integration_summary.csv"

    _plot(boss=boss, desi=desi, overall_status=overall, out_png=out_png, out_pdf=out_pdf)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "8.7.35.2 (BAO xi_l cross integration: BOSS+DESI)",
        "inputs": {
            "boss_metrics_json": str(boss_path).replace("\\", "/"),
            "desi_metrics_json": str(desi_path).replace("\\", "/"),
        },
        "integration": {
            "stage_a_boss": boss,
            "stage_b_desi": desi,
            "overall_status": overall,
            "decision_rule": {
                "stage_a": "|z|<=3:pass, 3<|z|<=5:watch, >5:reject",
                "stage_b": "promoted=true:pass, false:watch",
                "overall": "stage_a=reject -> reject; stage_a=pass & stage_b=pass -> pass; else watch",
            },
            "part4_scoreboard_update": part4_update,
        },
        "outputs": {
            "png": str(out_png).replace("\\", "/"),
            "pdf": str(out_pdf).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
            "summary_csv": str(out_csv).replace("\\", "/"),
        },
    }
    _write_json(out_json, payload)
    _write_csv(
        out_csv,
        rows=(
            ("stage_a_status", boss.get("status")),
            ("stage_a_max_abs_z", boss.get("max_abs_z")),
            ("stage_b_status", desi.get("status")),
            ("stage_b_promoted", desi.get("promoted")),
            ("stage_b_passing_tracers_n", desi.get("passing_tracers_n")),
            ("overall_status", overall),
            ("part4_scoreboard_update_recommend", bool(part4_update.get("recommend_update"))),
        ),
    )
    _copy_to_public((out_png, out_pdf, out_json, out_csv), out_public)

    print(f"[ok] png : {out_png}")
    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] public mirror: {out_public}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_xi_cross_integration",
                "argv": list(sys.argv),
                "inputs": {"boss_metrics": boss_path, "desi_metrics": desi_path},
                "outputs": {"png": out_png, "pdf": out_pdf, "metrics_json": out_json, "summary_csv": out_csv},
                "metrics": {
                    "stage_a_status": boss.get("status"),
                    "stage_a_max_abs_z": boss.get("max_abs_z"),
                    "stage_b_status": desi.get("status"),
                    "stage_b_promoted": desi.get("promoted"),
                    "overall_status": overall,
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
