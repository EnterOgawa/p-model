#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_redshift_pbg.py

宇宙膨張を仮定せず、背景の時間波密度 P_bg(t) の時間変化だけで赤方偏移を説明する
（P-model の宇宙論的な最小モデル）。

前提（添付メモの式）:
  P(x,t) = P_bg(t) P_local(x)
  ν_obs/ν_em = P_obs/P_em  →  1 + z = P_em/P_obs

低 z の近似:
  H0^(P) ≡ - (d/dt ln P_bg)|_{t0}
  z ≈ H0^(P) Δt ≈ H0^(P) D / c

出力（固定名）:
  - output/cosmology/cosmology_redshift_pbg.png
  - output/cosmology/cosmology_redshift_pbg_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_C = 299_792_458.0  # m/s
_MPC_M = 3.085_677_581_491_367_3e22  # m
_GYR_S = 1.0e9 * 365.25 * 24.0 * 3600.0


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


def _h0_si_from_km_s_mpc(h0_km_s_mpc: float) -> float:
    return (float(h0_km_s_mpc) * 1000.0) / _MPC_M


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cosmology: explain redshift by background time-wave density P_bg(t) (no expansion).",
    )
    parser.add_argument(
        "--h0",
        type=float,
        default=70.0,
        help="H0^(P) in km/s/Mpc for visualization (default: 70).",
    )
    parser.add_argument(
        "--lookback-gyr-max",
        type=float,
        default=15.0,
        help="Max lookback time for plots in Gyr (default: 15).",
    )
    args = parser.parse_args(argv)

    h0_km_s_mpc = float(args.h0)
    h0_si = _h0_si_from_km_s_mpc(h0_km_s_mpc)

    # Model choice: exponential background (constant H0^(P)) for the minimal demonstration.
    # Normalization: P_obs = 1 at t0.
    t_lb_gyr = np.linspace(0.0, float(args.lookback_gyr_max), 500)
    t_lb_s = t_lb_gyr * _GYR_S

    z_exact = np.exp(h0_si * t_lb_s) - 1.0
    z_low = h0_si * t_lb_s

    # Static (non-expanding) light-travel distance approximation.
    d_m = _C * t_lb_s
    d_gpc = d_m / (_MPC_M * 1000.0)

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    ax1.plot(t_lb_gyr, z_exact, label="P_bgモデル（指数）: z = exp(H0·Δt) − 1", linewidth=2.0)
    ax1.plot(t_lb_gyr, z_low, "--", label="低z近似: z ≈ H0·Δt", linewidth=1.5, alpha=0.9)
    ax1.set_title("赤方偏移 z とルックバック時間 Δt", fontsize=13)
    ax1.set_xlabel("ルックバック時間 Δt [Gyr]", fontsize=11)
    ax1.set_ylabel("赤方偏移 z", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(d_gpc, z_exact, label="P_bgモデル（指数）", linewidth=2.0)
    ax2.plot(d_gpc, z_low, "--", label="低z近似: z ≈ H0·D/c", linewidth=1.5, alpha=0.9)
    ax2.set_title("赤方偏移 z と距離 D（静的近似 D≈cΔt）", fontsize=13)
    ax2.set_xlabel("距離 D [Gpc]", fontsize=11)
    ax2.set_ylabel("赤方偏移 z", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "宇宙論：宇宙膨張なしで赤方偏移を P で説明（背景Pの時間変化）",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        f"H0^(P) = - (d/dt ln P_bg)|t0 = {h0_km_s_mpc:.1f} km/s/Mpc（可視化用の代表値）",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cosmology_redshift_pbg.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "P_factorization": "P(x,t)=P_bg(t) P_local(x)",
            "redshift_relation": "1+z = P_em/P_obs",
            "P_bg_model": "P_bg(t) ∝ exp(-H0^(P) (t-t0))",
            "H0P_definition": "H0^(P) = - (d/dt ln P_bg)|t0",
        },
        "params": {"H0P_km_s_Mpc": h0_km_s_mpc, "lookback_gyr_max": float(args.lookback_gyr_max)},
        "derived": {"H0P_SI_s^-1": h0_si},
        "outputs": {"png": str(png_path)},
        "notes": [
            "この図は“宇宙膨張なしでも赤方偏移が出る”という機構の例示であり、宇宙論全体（CMB/元素合成など）を単独で置き換える主張ではない。",
            "差分予測（ΛCDMとの比較）を行う場合は、距離-時間の扱い（静的近似の限界）と観測の定義（D_L 等）を明確化して追加する。",
        ],
    }
    json_path = out_dir / "cosmology_redshift_pbg_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {png_path}")
    print(f"[ok] json: {json_path}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_redshift_pbg",
                "argv": sys.argv,
                "metrics": {"H0P_km_s_Mpc": h0_km_s_mpc, "lookback_gyr_max": float(args.lookback_gyr_max)},
                "outputs": {"png": png_path, "metrics_json": json_path},
            }
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

