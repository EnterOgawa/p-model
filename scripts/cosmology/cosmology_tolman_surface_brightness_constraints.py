#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_tolman_surface_brightness_constraints.py

Tolman表面輝度テスト（SB dimming）の一次ソース制約を固定入力として取り込み、
「棄却条件（どの精度なら棄却できるか / 既に棄却されるか）」の形で固定出力する。

入力（固定）:
  - data/cosmology/tolman_surface_brightness_constraints.json

定義（一次ソースのパラメータ化）:
  SB(z)/SB(0) ∝ (1+z)^(-n)
  （magnitude表現では 2.5 log(1+z)^n）

モデル予測（本リポジトリの最小仮定）:
  - FRW（Tolman）: n=4
  - 背景P（膨張なし・静的幾何 + 赤方偏移（光子エネルギー）+ 時間伸長）: n=2

注意：
  - 実際の観測から推定される n には、銀河の光度進化（evolution）が混入する。
    ここでは一次ソースが提示する n を固定入力として扱い、
    (i) 純粋スケーリングとの差（zスコア）、
    (ii) 整合に必要な「進化指数」の符号（n_model - n_obs）
    を比較する。

出力（固定名）:
  - output/cosmology/cosmology_tolman_surface_brightness_constraints.png
  - output/cosmology/cosmology_tolman_surface_brightness_constraints_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


@dataclass(frozen=True)
class Constraint:
    id: str
    short_label: str
    title: str
    n_obs: float
    n_sigma: float
    sigma_note: str
    redshift_range_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "Constraint":
        return Constraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            n_obs=float(j["n_obs"]),
            n_sigma=float(j["n_sigma"]),
            sigma_note=str(j.get("sigma_note") or ""),
            redshift_range_note=str(j.get("redshift_range_note") or ""),
            source=dict(j.get("source") or {}),
        )


def compute(rows: Sequence[Constraint]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # Model predictions for n
    n_frw = 4.0
    n_pbg_static = 2.0

    for r in rows:
        sig = float(r.n_sigma)
        z_frw = None
        z_pbg = None
        if sig > 0:
            z_frw = (n_frw - float(r.n_obs)) / sig
            z_pbg = (n_pbg_static - float(r.n_obs)) / sig

        # "Non-rejection" threshold (3σ): require sig >= |n_model - n_obs| / 3.
        sig_need_nonreject_pbg_3sigma = None
        if sig > 0:
            sig_need_nonreject_pbg_3sigma = abs(n_pbg_static - float(r.n_obs)) / 3.0

        out.append(
            {
                "id": r.id,
                "short_label": r.short_label,
                "title": r.title,
                "n_obs": float(r.n_obs),
                "n_sigma": sig,
                "n_pred_frw": n_frw,
                "n_pred_pbg_static": n_pbg_static,
                "z_frw": None if z_frw is None else float(z_frw),
                "z_pbg_static": None if z_pbg is None else float(z_pbg),
                "evolution_exponent_needed_frw": float(n_frw - float(r.n_obs)),
                "evolution_exponent_needed_pbg_static": float(n_pbg_static - float(r.n_obs)),
                "sigma_needed_to_not_reject_pbg_static_3sigma": (
                    None if sig_need_nonreject_pbg_3sigma is None else float(sig_need_nonreject_pbg_3sigma)
                ),
                "sigma_note": r.sigma_note,
                "redshift_range_note": r.redshift_range_note,
                "source": r.source,
            }
        )
    return out


def _plot(rows: Sequence[Dict[str, Any]], *, out_png: Path, z_max: float) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    z = np.linspace(0.0, z_max, 500)
    one_p_z = 1.0 + z

    # Model curves (normalized at z=0)
    sb_frw = one_p_z**-4
    sb_pbg = one_p_z**-2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel 1: SB dimming curves + observational exponent band(s)
    ax1.plot(z, sb_frw, label="FRW: SB ∝ (1+z)^-4", linewidth=2.0)
    ax1.plot(z, sb_pbg, label="背景P（静的）: SB ∝ (1+z)^-2", linewidth=2.0)

    colors = ["#444444", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, r in enumerate(rows):
        n_obs = _safe_float(r.get("n_obs"))
        sig = _safe_float(r.get("n_sigma"))
        label = str(r.get("short_label") or r.get("id") or "")
        if n_obs is None or sig is None or sig <= 0:
            continue
        sb_mid = one_p_z ** (-float(n_obs))
        sb_lo = one_p_z ** (-(float(n_obs) + float(sig)))
        sb_hi = one_p_z ** (-(float(n_obs) - float(sig)))
        ax1.fill_between(z, sb_lo, sb_hi, alpha=0.18, color=colors[i % len(colors)], label=f"観測制約（1σ, {label}）")
        ax1.plot(z, sb_mid, color=colors[i % len(colors)], linewidth=1.0, alpha=0.8)

    ax1.set_title("Tolman表面輝度（正規化）：SB(z)/SB(0)", fontsize=13)
    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("SB(z)/SB(0)", fontsize=11)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: exponent summary
    y = np.arange(len(rows), dtype=float)
    n_obs = np.array([float(r.get("n_obs", float("nan"))) for r in rows], dtype=float)
    n_sig = np.array([float(r.get("n_sigma", float("nan"))) for r in rows], dtype=float)
    labels = [str(r.get("short_label") or r.get("id") or "") for r in rows]

    ax2.axvline(4.0, color="#333333", linewidth=1.2, alpha=0.85, label="FRW（Tolman）: n=4")
    ax2.axvline(2.0, color="#d62728", linewidth=1.2, alpha=0.85, label="背景P（静的）: n=2")
    ax2.errorbar(
        n_obs,
        y,
        xerr=np.where(np.isfinite(n_sig) & (n_sig > 0), n_sig, 0.0),
        fmt="o",
        capsize=4,
        color="#1f77b4",
        ecolor="#1f77b4",
        label="観測（一次ソースの公表値）",
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("n（SB ∝ (1+z)^-n）", fontsize=11)
    ax2.set_title("有効指数 n の観測値とモデル予測（進化が系統）", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle("宇宙論（棄却条件の補強）：Tolman表面輝度テスト（一次ソース）", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "注：観測から推定される n には銀河の光度進化が混入する。ここでは一次ソースの n を固定入力として扱い、差の符号/大きさを比較する。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: Tolman surface brightness constraints and rejection condition.")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "tolman_surface_brightness_constraints.json"),
        help="Input JSON (default: data/cosmology/tolman_surface_brightness_constraints.json)",
    )
    ap.add_argument(
        "--z-max",
        type=float,
        default=1.3,
        help="Max redshift for plotting SB(z) (default: 1.3).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    z_max = float(args.z_max)
    if not (z_max > 0.0):
        raise ValueError("--z-max must be > 0")

    src = _read_json(data_path)
    constraints = [Constraint.from_json(c) for c in (src.get("constraints") or [])]
    if not constraints:
        raise SystemExit(f"no constraints found in: {data_path}")

    rows = compute(constraints)

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cosmology_tolman_surface_brightness_constraints.png"
    _plot(rows, out_png=png_path, z_max=z_max)

    out_json = out_dir / "cosmology_tolman_surface_brightness_constraints_metrics.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "definition": src.get("definition") or {},
        "assumptions": {
            "parameterization": "SB(z)/SB(0) ∝ (1+z)^(-n)",
            "criterion_example": "|n_obs - n_model| > 3σ => reject (rule-of-thumb, if interpreted literally)",
            "note": "n_obs includes luminosity evolution systematics; use z-score and evolution_exponent_needed mainly as a sign/scale check.",
        },
        "rows": rows,
        "params": {"z_max": z_max},
        "outputs": {"png": str(png_path).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {png_path}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_tolman_surface_brightness_constraints",
                "argv": list(sys.argv),
                "inputs": {"data": data_path},
                "outputs": {"png": png_path, "metrics_json": out_json},
                "metrics": {"z_max": z_max, "n_constraints": len(rows)},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

