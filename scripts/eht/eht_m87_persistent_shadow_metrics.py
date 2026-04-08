#!/usr/bin/env python3
"""
目的: EHT topic の eht m87 persistent shadow metrics に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.utils.plot_style import (  # noqa: E402
    apply_paper_style,
    apply_wavep_figure_layout,
    get_wavep_font_size,
    resolve_wavep_cjk_font_family,
)


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = resolve_wavep_cjk_font_family(preferred_name="Noto Sans CJK JP")
        # 条件分岐: `preferred` を満たす経路を評価する。
        if preferred:
            mpl.rcParams["font.family"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return

        fallback = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in fallback if name in available]
        # 条件分岐: `chosen` を満たす経路を評価する。
        if chosen:
            mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: `_save_public_figure` の入出力契約と処理意図を定義する。

def _save_public_figure(
    fig: Any,
    *,
    out_dir: Path,
    public_dir: Path,
    stem: str,
    dpi_png: int = 220,
) -> Tuple[Path, Path, Path, Path]:
    import matplotlib.pyplot as plt

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    public_png_path = public_dir / f"{stem}.png"
    public_pdf_path = public_dir / f"{stem}.pdf"
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(png_path, dpi=int(dpi_png))
        fig.savefig(pdf_path)

    shutil.copy2(png_path, public_png_path)
    shutil.copy2(pdf_path, public_pdf_path)
    return png_path, pdf_path, public_png_path, public_pdf_path


# 関数: `_sigma_sym` の入出力契約と処理意図を定義する。

def _sigma_sym(sigma_minus: float, sigma_plus: float, *, mode: str) -> float:
    sm = float(sigma_minus)
    sp = float(sigma_plus)
    # 条件分岐: `not (math.isfinite(sm) and math.isfinite(sp) and sm >= 0 and sp >= 0)` を満たす経路を評価する。
    if not (math.isfinite(sm) and math.isfinite(sp) and sm >= 0 and sp >= 0):
        return float("nan")

    # 条件分岐: `mode == "avg"` を満たす経路を評価する。

    if mode == "avg":
        return 0.5 * (sm + sp)

    # 条件分岐: `mode == "max"` を満たす経路を評価する。

    if mode == "max":
        return max(sm, sp)

    return float("nan")


# クラス: `RingMeasurement` の責務と境界条件を定義する。

@dataclass(frozen=True)
class RingMeasurement:
    epoch: str
    label: str
    diameter_uas: float
    sigma_minus_uas: float
    sigma_plus_uas: float
    source_key: str


# 関数: `_extract_m87_ring_measurements` の入出力契約と処理意図を定義する。

def _extract_m87_ring_measurements(eht: Dict[str, Any]) -> List[RingMeasurement]:
    objects = eht.get("objects") if isinstance(eht.get("objects"), list) else []
    m87 = None
    for o in objects:
        # 条件分岐: `not isinstance(o, dict)` を満たす経路を評価する。
        if not isinstance(o, dict):
            continue

        # 条件分岐: `str(o.get("key") or "").strip() == "m87"` を満たす経路を評価する。

        if str(o.get("key") or "").strip() == "m87":
            m87 = o
            break

    # 条件分岐: `not isinstance(m87, dict)` を満たす経路を評価する。

    if not isinstance(m87, dict):
        raise RuntimeError("data/eht/eht_black_holes.json: object key=m87 not found")

    ms = m87.get("ring_measurements")
    # 条件分岐: `not isinstance(ms, list) or not ms` を満たす経路を評価する。
    if not isinstance(ms, list) or not ms:
        raise RuntimeError("data/eht/eht_black_holes.json: m87.ring_measurements is missing/empty")

    out: List[RingMeasurement] = []
    for item in ms:
        # 条件分岐: `not isinstance(item, dict)` を満たす経路を評価する。
        if not isinstance(item, dict):
            continue

        epoch = str(item.get("epoch") or "").strip()
        label = str(item.get("label") or "").strip() or epoch
        source_key = str(item.get("source_key") or "").strip()
        # 条件分岐: `not (epoch and source_key)` を満たす経路を評価する。
        if not (epoch and source_key):
            continue

        d = float(item.get("diameter_uas", float("nan")))
        sm = float(item.get("sigma_minus_uas", float("nan")))
        sp = float(item.get("sigma_plus_uas", float("nan")))
        # 条件分岐: `not (math.isfinite(d) and math.isfinite(sm) and math.isfinite(sp))` を満たす経路を評価する。
        if not (math.isfinite(d) and math.isfinite(sm) and math.isfinite(sp)):
            continue

        # 条件分岐: `sm < 0 or sp < 0` を満たす経路を評価する。

        if sm < 0 or sp < 0:
            continue

        out.append(
            RingMeasurement(
                epoch=epoch,
                label=label,
                diameter_uas=d,
                sigma_minus_uas=sm,
                sigma_plus_uas=sp,
                source_key=source_key,
            )
        )

    # 条件分岐: `len(out) < 2` を満たす経路を評価する。

    if len(out) < 2:
        raise RuntimeError("m87.ring_measurements: expected >=2 valid entries")

    return out


# 関数: `_find_epoch` の入出力契約と処理意図を定義する。

def _find_epoch(measurements: List[RingMeasurement], epoch: str) -> RingMeasurement:
    for m in measurements:
        # 条件分岐: `m.epoch == epoch` を満たす経路を評価する。
        if m.epoch == epoch:
            return m

    raise RuntimeError(f"missing measurement for epoch={epoch!r}")


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    inp = root / "data" / "eht" / "eht_black_holes.json"
    out_dir = root / "output" / "private" / "eht"
    public_dir = root / "output" / "public" / "eht"
    out_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    eht = _read_json(inp)
    ms = _extract_m87_ring_measurements(eht)

    m2017 = _find_epoch(ms, "2017")
    m2018 = _find_epoch(ms, "2018")

    delta = float(m2018.diameter_uas - m2017.diameter_uas)

    sig2017_avg = _sigma_sym(m2017.sigma_minus_uas, m2017.sigma_plus_uas, mode="avg")
    sig2018_avg = _sigma_sym(m2018.sigma_minus_uas, m2018.sigma_plus_uas, mode="avg")
    sig2017_max = _sigma_sym(m2017.sigma_minus_uas, m2017.sigma_plus_uas, mode="max")
    sig2018_max = _sigma_sym(m2018.sigma_minus_uas, m2018.sigma_plus_uas, mode="max")

    sig_delta_avg = math.sqrt(sig2017_avg**2 + sig2018_avg**2) if (math.isfinite(sig2017_avg) and math.isfinite(sig2018_avg)) else float("nan")
    sig_delta_max = math.sqrt(sig2017_max**2 + sig2018_max**2) if (math.isfinite(sig2017_max) and math.isfinite(sig2018_max)) else float("nan")

    z_delta_avg = (delta / sig_delta_avg) if (math.isfinite(delta) and math.isfinite(sig_delta_avg) and sig_delta_avg > 0) else float("nan")
    z_delta_max = (delta / sig_delta_max) if (math.isfinite(delta) and math.isfinite(sig_delta_max) and sig_delta_max > 0) else float("nan")

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(inp.relative_to(root)).replace("\\", "/"),
        "target": "M87*",
        "ring_measurements": [
            {
                "epoch": m.epoch,
                "label": m.label,
                "diameter_uas": m.diameter_uas,
                "sigma_minus_uas": m.sigma_minus_uas,
                "sigma_plus_uas": m.sigma_plus_uas,
                "source_key": m.source_key,
            }
            for m in ms
        ],
        "delta_2018_minus_2017_uas": delta,
        "delta_sigma_uas_avg_sym": sig_delta_avg,
        "delta_sigma_uas_max_sym": sig_delta_max,
        "delta_z_avg_sym": z_delta_avg,
        "delta_z_max_sym": z_delta_max,
        "notes": [
            "2018 ring diameter is compared against 2017 as an independent-epoch reproducibility check.",
            "Asymmetric uncertainties are symmetrized either by average (avg_sym) or conservative max (max_sym) to form a simple z-score proxy.",
        ],
    }

    out_json = out_dir / "eht_m87_persistent_shadow_ring_diameter_metrics.json"
    _write_json(out_json, payload)

    # Plot (epoch vs diameter with asymmetric error bars).
    png_path: Optional[Path] = None
    public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        apply_paper_style()
        _set_japanese_font()

        epochs = [m2017.epoch, m2018.epoch]
        labels = [
            "2017",
            "2018",
        ]
        y = np.array([m2017.diameter_uas, m2018.diameter_uas], dtype=float)
        yerr_minus = np.array([m2017.sigma_minus_uas, m2018.sigma_minus_uas], dtype=float)
        yerr_plus = np.array([m2017.sigma_plus_uas, m2018.sigma_plus_uas], dtype=float)
        x = np.arange(len(epochs), dtype=float)

        fig, ax = plt.subplots()
        apply_wavep_figure_layout(fig, template="part2_single_panel")
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([yerr_minus, yerr_plus]),
            fmt="o",
            color="#d62728",
            ecolor="#d62728",
            elinewidth=2.0,
            capsize=5,
            label="観測リング直径 θ_ring（M87*）",
            zorder=3,
        )
        ax.plot(x, y, color="#d62728", alpha=0.35, lw=2.0, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("角直径 [µas]")
        ax.set_title("EHT M87*: リング直径の multi-epoch 整合（2017 vs 2018）", pad=6.0)
        ax.tick_params(axis="x", pad=6)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlim(-0.18, 1.18)

        # Annotate delta.
        if math.isfinite(delta):
            ax.text(
                0.98,
                0.92,
                f"Δ(2018−2017) = {delta:+.1f} µas\nz(avg/max)={z_delta_avg:.2f} / {z_delta_max:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=get_wavep_font_size("note"),
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
            )

        ax.legend(loc="upper left", framealpha=0.9, fontsize=get_wavep_font_size("legend"))
        fig.subplots_adjust(bottom=0.14, right=0.95)
        png_path, pdf_path, public_png_path, public_pdf_path = _save_public_figure(
            fig,
            out_dir=out_dir,
            public_dir=public_dir,
            stem="eht_m87_persistent_shadow_ring_diameter",
        )
        plt.close(fig)
    except Exception as e:
        print(f"[warn] plot skipped: {e}")
        png_path = None
        public_png_path = None

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_m87_persistent_shadow_metrics",
                "inputs": {"eht_black_holes": str(inp.relative_to(root)).replace("\\", "/")},
                "outputs": {
                    "json": str(out_json.relative_to(root)).replace("\\", "/"),
                    "png": str(png_path.relative_to(root)).replace("\\", "/") if isinstance(png_path, Path) else None,
                    "png_public": (
                        str(public_png_path.relative_to(root)).replace("\\", "/") if isinstance(public_png_path, Path) else None
                    ),
                },
                "metrics": {"delta_uas": delta, "delta_z_avg_sym": z_delta_avg, "delta_z_max_sym": z_delta_max},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    # 条件分岐: `isinstance(png_path, Path)` を満たす経路を評価する。
    if isinstance(png_path, Path):
        print(f"[ok] png : {png_path}")

    # 条件分岐: `isinstance(public_png_path, Path)` を満たす経路を評価する。

    if isinstance(public_png_path, Path):
        print(f"[ok] png : {public_png_path}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
