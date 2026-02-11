#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cmb_temperature_scaling_constraints.py

Step 14.2.3（独立プローブで相互検証）:
CMB温度スケーリング T(z) の一次ソース制約を固定出力する。

目的：
- 距離指標（D_L/D_A）と独立な観測で、赤方偏移に伴う温度スケーリングが標準（T∝1+z）と整合するか確認する。
- DDR（距離二重性）の“回復”をエネルギー側の指数だけで行う逃げ道（例：p_T≈3）を排除できるかを示す。

パラメータ化（一次ソースの一般形）：
  T(z) = T0 (1+z)^(1-β_T)  （標準は β_T=0）
等価に
  T(z) = T0 (1+z)^(p_T)  とおけば p_T = 1 - β_T

入力（固定）:
  - data/cosmology/cmb_temperature_scaling_constraints.json

出力（固定名）:
  - output/cosmology/cosmology_cmb_temperature_scaling_constraints.png
  - output/cosmology/cosmology_cmb_temperature_scaling_constraints_metrics.json
"""

from __future__ import annotations

import argparse
import json
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


def _fmt_float(x: float, *, digits: int = 6) -> str:
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Constraint:
    id: str
    short_label: str
    title: str
    beta_T: float
    beta_T_sigma: float
    sigma_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "Constraint":
        return Constraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            beta_T=float(j["beta_T"]),
            beta_T_sigma=float(j["beta_T_sigma"]),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


def compute(rows: Sequence[Constraint]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # Convert β_T -> p_T (T(z)=T0(1+z)^(p_T))
    # Benchmarks:
    # - Standard adiabatic scaling: p_T=1 (β_T=0)
    # - No scaling (static CMB temperature): p_T=0 (β_T=1)
    # - If one tried to recover DDR (ε0≈0) only by changing the energy exponent with no opacity/evolution,
    #   p_T≈3 is a useful scale marker (same algebraic role as p_e or p_t in ε0_model).
    p_T_std = 1.0
    p_T_no_scaling = 0.0
    p_T_ddr_fix_only = 3.0

    for r in rows:
        sig = float(r.beta_T_sigma)
        if not (sig > 0.0):
            raise ValueError(f"beta_T_sigma must be >0: {r.id}")

        p_T_obs = 1.0 - float(r.beta_T)
        p_T_sigma = sig

        def z(model_pt: float) -> float:
            return (model_pt - p_T_obs) / p_T_sigma

        out.append(
            {
                "id": r.id,
                "short_label": r.short_label,
                "title": r.title,
                "beta_T_obs": float(r.beta_T),
                "beta_T_sigma": sig,
                "p_T_obs": float(p_T_obs),
                "p_T_sigma": float(p_T_sigma),
                "z_std": z(p_T_std),
                "z_no_scaling": z(p_T_no_scaling),
                "z_ddr_fix_only": z(p_T_ddr_fix_only),
                "benchmarks": {
                    "p_T_std": p_T_std,
                    "p_T_no_scaling": p_T_no_scaling,
                    "p_T_ddr_fix_only": p_T_ddr_fix_only,
                },
                "sigma_note": r.sigma_note,
                "source": r.source,
            }
        )

    return out


def _plot(rows: Sequence[Dict[str, Any]], *, out_png: Path) -> None:
    labels = [str(r.get("short_label") or r.get("id") or "") for r in rows]
    y = np.arange(len(rows))[::-1]

    p_obs = np.array([float(r["p_T_obs"]) for r in rows], dtype=float)
    p_sig = np.array([float(r["p_T_sigma"]) for r in rows], dtype=float)

    z_std = np.array([float(r["z_std"]) for r in rows], dtype=float)
    z_ns = np.array([float(r["z_no_scaling"]) for r in rows], dtype=float)
    z_ddr = np.array([float(r["z_ddr_fix_only"]) for r in rows], dtype=float)

    p_std = float(rows[0]["benchmarks"]["p_T_std"]) if rows else 1.0
    p_ns = float(rows[0]["benchmarks"]["p_T_no_scaling"]) if rows else 0.0
    p_ddr = float(rows[0]["benchmarks"]["p_T_ddr_fix_only"]) if rows else 3.0

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    # Left: z-scores
    ax1.axvline(0.0, color="k", linewidth=1.0, alpha=0.6)
    ax1.axvline(-3.0, color="#999999", linewidth=1.0, alpha=0.7, linestyle="--")
    ax1.axvline(3.0, color="#999999", linewidth=1.0, alpha=0.7, linestyle="--")

    dy = 0.12
    ax1.scatter(z_std, y + dy, s=45, label=f"標準: p_T={_fmt_float(p_std, digits=3)}", color="#1f77b4")
    ax1.scatter(z_ns, y, s=45, label=f"温度一定: p_T={_fmt_float(p_ns, digits=3)}", color="#d62728")
    ax1.scatter(z_ddr, y - dy, s=45, label=f"DDRをp_Tだけで回復: p_T={_fmt_float(p_ddr, digits=3)}", color="#2ca02c")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("z-score（(p_T_model - p_T_obs)/σ）", fontsize=11)
    ax1.set_title("T(z)スケーリング：モデルのz-score", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax1.legend(fontsize=9, loc="lower right")

    # Right: observed p_T with reference lines
    ax2.axvline(p_std, color="#1f77b4", linewidth=1.2, alpha=0.85, label="標準: p_T=1（β_T=0）")
    ax2.axvline(p_ns, color="#d62728", linewidth=1.2, alpha=0.85, label="温度一定: p_T=0（β_T=1）")
    ax2.axvline(p_ddr, color="#2ca02c", linewidth=1.2, alpha=0.85, label="DDR回復（p_Tのみ）: p_T=3")
    ax2.errorbar(
        p_obs,
        y,
        xerr=p_sig,
        fmt="o",
        capsize=4,
        color="#111111",
        ecolor="#111111",
        label="観測（一次ソースの公表値）",
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("p_T（T(z)=T0(1+z)^(p_T)）", fontsize=11)
    ax2.set_title("観測 p_T（CMB温度スケーリング指数）の一次ソース制約", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle("宇宙論（独立プローブ）：CMB温度スケーリング T(z) の制約", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "注：T(z)は距離指標と独立に推定できる。標準スケーリング（p_T=1）と整合するため、p_Tの変更だけでDDRを回復する説明は成立しない。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: CMB temperature scaling constraints (T(z)).")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "cmb_temperature_scaling_constraints.json"),
        help="Input JSON (default: data/cosmology/cmb_temperature_scaling_constraints.json)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    src = _read_json(data_path)
    constraints = [Constraint.from_json(c) for c in (src.get("constraints") or [])]
    if not constraints:
        raise SystemExit(f"no constraints found in: {data_path}")

    rows = compute(constraints)

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cosmology_cmb_temperature_scaling_constraints.png"
    _plot(rows, out_png=png_path)

    out_json = out_dir / "cosmology_cmb_temperature_scaling_constraints_metrics.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "definition": src.get("definition") or {},
        "assumptions": {
            "parameterization": "T(z)=T0(1+z)^(1-β_T)  (=> p_T=1-β_T)",
            "benchmarks": {
                "standard": {"p_T": 1.0, "beta_T": 0.0},
                "no_scaling": {"p_T": 0.0, "beta_T": 1.0},
                "DDR_fix_only": {"p_T": 3.0, "note": "opacity/evolutionなしでε0=0を満たすためのスケール目安"},
            },
        },
        "rows": rows,
        "outputs": {"png": str(png_path).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {png_path}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_cmb_temperature_scaling_constraints",
                "argv": list(sys.argv),
                "inputs": {"data": data_path},
                "outputs": {"png": png_path, "metrics_json": out_json},
                "metrics": {"n_constraints": len(rows)},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
