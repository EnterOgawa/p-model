#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_observable_scalings.py

宇宙膨張（FRW/ΛCDMの一般論）と、「宇宙膨張なし＋背景Pの時間変化で赤方偏移」を置く場合で、
観測量のスケーリングにどのような“差”が出るかを可視化する（差分予測の入口）。

目的：
- 「赤方偏移は膨張以外でも出る」だけでは議論が曖昧になるため、
  観測で判別できる形（距離二重性 / Tolman表面輝度 など）に結びつける。

前提（最小仮定）：
- 赤方偏移の定義： 1+z = ν_em / ν_obs
- エネルギー：E_ph ∝ 1/(1+z)
- 時間伸長：Δt_obs = (1+z) Δt_em
  - 注：背景Pモデルでは ν_obs/ν_em=P_obs/P_em を仮定するため、時間スケールも同じ比で伸長し、
    最小予測として p_t=1（Δt_obs=(1+z)Δt_em）が自然に出る（SNスペクトル年齢とも整合）。

比較するモデル：
1) FRW（光子保存＋幾何学的膨張）:
   - 距離二重性（Etherington）： D_L = (1+z)^2 D_A  →  η(z)≡D_L/((1+z)^2 D_A)=1
   - Tolman表面輝度： SB ∝ (1+z)^-4

2) 背景P（宇宙膨張なし・静的幾何）:
   - 幾何学的には D_A = r（Euclid）
   - 赤方偏移（E_ph）と時間伸長（到達率）だけで減光 → F ∝ 1/(1+z)^2
   - よって D_L = r (1+z)  →  D_L = (1+z) D_A
     → η(z)=1/(1+z)
   - 表面輝度は SB ∝ (1+z)^-2（光子エネルギー×到達率のみ）

出力（固定名）:
  - output/cosmology/cosmology_observable_scalings.png
  - output/cosmology/cosmology_observable_scalings_metrics.json
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cosmology: observable scaling differences (FRW vs no-expansion background-P redshift).",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=3.0,
        help="Max redshift for plots (default: 3.0).",
    )
    args = parser.parse_args(argv)

    z_max = float(args.z_max)
    if not (z_max > 0.0):
        raise ValueError("--z-max must be > 0")

    z = np.linspace(0.0, z_max, 800)
    one_p_z = 1.0 + z

    # Distance duality function η(z) ≡ D_L / ((1+z)^2 D_A)
    eta_frw = np.ones_like(z)
    eta_pbg_static = 1.0 / one_p_z

    # Tolman surface brightness scaling (normalized at z=0)
    sb_frw = one_p_z**-4
    sb_pbg_static = one_p_z**-2

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    ax1.plot(z, eta_frw, label="FRW: η(z)=1（距離二重性）", linewidth=2.0)
    ax1.plot(z, eta_pbg_static, label="背景P（静的）: η(z)=1/(1+z)", linewidth=2.0)
    ax1.set_title("距離二重性 η(z) = D_L / ((1+z)^2 D_A)", fontsize=13)
    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("η(z)", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(z, sb_frw, label="FRW: SB ∝ (1+z)^-4", linewidth=2.0)
    ax2.plot(z, sb_pbg_static, label="背景P（静的）: SB ∝ (1+z)^-2", linewidth=2.0)
    ax2.set_title("Tolman表面輝度（正規化）：SB(z)/SB(0)", fontsize=13)
    ax2.set_xlabel("赤方偏移 z", fontsize=11)
    ax2.set_ylabel("SB(z)/SB(0)", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "宇宙論（差分予測の入口）：膨張（FRW） vs 背景P（膨張なし）の観測量スケーリング",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "注：背景Pでは周波数比（1+z）がそのまま時間比に出るため、p_t=1（Δt_obs=(1+z)Δt_em）が自然に出る。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cosmology_observable_scalings.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "scope": "differential predictions (scalings), no data fit",
        "assumptions": {
            "redshift_definition": "1+z = nu_em/nu_obs",
            "photon_energy_scaling": "E_ph ∝ 1/(1+z)",
            "time_dilation_scaling": "Δt_obs = (1+z) Δt_em",
        },
        "models": {
            "FRW": {
                "distance_duality": "D_L = (1+z)^2 D_A  (=> η=1)",
                "tolman_surface_brightness": "SB ∝ (1+z)^-4",
            },
            "P_bg_static": {
                "geometry": "no expansion (Euclid): D_A = r",
                "luminosity_distance": "D_L = r (1+z)  (=> D_L = (1+z) D_A, η=1/(1+z))",
                "tolman_surface_brightness": "SB ∝ (1+z)^-2",
            },
        },
        "params": {"z_max": z_max},
        "outputs": {"png": str(png_path)},
        "notes": [
            "距離二重性やTolmanテストは、赤方偏移の起源（膨張 vs 別機構）を“観測量”として分ける候補になる。",
            "本図はデータに当てた結果ではなく、定義と最小仮定から導いた差分スケーリングの可視化である。",
        ],
    }
    json_path = out_dir / "cosmology_observable_scalings_metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {png_path}")
    print(f"[ok] json: {json_path}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_observable_scalings",
                "argv": sys.argv,
                "metrics": {"z_max": z_max},
                "outputs": {"png": png_path, "metrics_json": json_path},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
