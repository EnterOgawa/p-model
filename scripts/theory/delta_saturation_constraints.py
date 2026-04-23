#!/usr/bin/env python3
"""
目的: 理論 topic の delta saturation constraints に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.utils.figure_locale_paths import localize_figure_output_path, resolve_figure_output_locale  # noqa: E402
from scripts.utils.plot_style import (  # noqa: E402
    apply_wavep_figure_layout,
    get_wavep_font_size,
    install_wavep_font_profile,
    resolve_wavep_cjk_font_family,
)

FIGURE_LOCALE = resolve_figure_output_locale()
IS_EN = FIGURE_LOCALE == "en"


# 関数: `_t` の入出力契約と処理意図を定義する。
def _t(ja: str, en: str) -> str:
    return en if IS_EN else ja

# 関数: `_repo_root` の入出力契約と処理意図を定義する。

def _repo_root() -> Path:
    return _ROOT


# 関数: `_configure_font` の入出力契約と処理意図を定義する。

def _configure_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        install_wavep_font_profile(profile_name="part2_astrophysics")
        if IS_EN:
            mpl.rcParams["font.family"] = ["DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return

        preferred = resolve_wavep_cjk_font_family(preferred_name="Noto Sans CJK JP")
        if preferred:
            mpl.rcParams["font.family"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return

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


# 関数: `_gamma_max` の入出力契約と処理意図を定義する。

def _gamma_max(delta: float) -> float:
    # gamma_max = sqrt((1+δ)/δ)  (≈ 1/sqrt(δ) when δ<<1)
    if not (math.isfinite(delta) and delta > 0):
        return float("nan")

    return math.sqrt((1.0 + float(delta)) / float(delta))


# 関数: `_delta_upper_for_gamma` の入出力契約と処理意図を定義する。

def _delta_upper_for_gamma(gamma_obs: float) -> float:
    # Require gamma_max >= gamma_obs
    # => (1+δ)/δ >= gamma^2  -> 1/δ + 1 >= gamma^2 -> δ <= 1/(gamma^2 - 1)
    if not (math.isfinite(gamma_obs) and gamma_obs > 1.0):
        return float("nan")

    return 1.0 / (float(gamma_obs) * float(gamma_obs) - 1.0)


# 関数: `_fmt_sci` の入出力契約と処理意図を定義する。

def _fmt_sci(x: float, *, digits: int = 2) -> str:
    # 条件分岐: `not math.isfinite(x)` を満たす経路を評価する。
    if not math.isfinite(x):
        return "n/a"

    # 条件分岐: `x == 0.0` を満たす経路を評価する。

    if x == 0.0:
        return "0"

    return f"{x:.{digits}e}"


# クラス: `Example` の責務と境界条件を定義する。

@dataclass(frozen=True)
class Example:
    key: str
    label: str
    energy_ev: float
    rest_mass_ev: float
    notes: str
    source: str


# 関数: `_parse_example` の入出力契約と処理意図を定義する。

def _parse_example(o: Dict[str, Any]) -> Example:
    return Example(
        key=str(o.get("key") or ""),
        label=str(o.get("label") or ""),
        energy_ev=float(o["energy_ev"]),
        rest_mass_ev=float(o["rest_mass_ev"]),
        notes=str(o.get("notes") or ""),
        source=str(o.get("source") or ""),
    )


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    default_in_path = root / "data" / "theory" / "delta_saturation_examples.json"
    default_out_dir = root / "output" / "theory"
    default_private_out_dir = root / "output" / "private" / "theory"
    default_public_out_dir = root / "output" / "public" / "theory"

    ap = argparse.ArgumentParser(description="Constraint chart for the saturation constant δ (Phase 4 differential).")
    ap.add_argument("--input", type=str, default=str(default_in_path), help="Input JSON (default: data/theory/...)")
    ap.add_argument(
        "--outdir", type=str, default=str(default_out_dir), help="Output directory (default: output/theory)"
    )
    ap.add_argument(
        "--private-outdir", type=str, default=str(default_private_out_dir), help="Private mirror directory."
    )
    ap.add_argument(
        "--public-outdir", type=str, default=str(default_public_out_dir), help="Public mirror directory."
    )
    ap.add_argument(
        "--delta",
        type=float,
        default=float("nan"),
        help="Adopted δ (override the value in the input JSON).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    private_out_dir = Path(args.private_outdir)
    public_out_dir = Path(args.public_outdir)
    if IS_EN:
        out_dir = private_out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    private_out_dir.mkdir(parents=True, exist_ok=True)
    public_out_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `not in_path.exists()` を満たす経路を評価する。
    if not in_path.exists():
        print(f"[err] missing input: {in_path}")
        return 2

    ref = _read_json(in_path)
    ref_delta = float(((ref.get("pmodel") or {}).get("delta_adopted")) or 1e-60)
    delta_adopted = float(args.delta) if math.isfinite(float(args.delta)) else ref_delta

    objs = ref.get("examples") or []
    examples = [_parse_example(o) for o in objs if isinstance(o, dict)]
    # 条件分岐: `not examples` を満たす経路を評価する。
    if not examples:
        print("[err] no examples in input json")
        return 2

    gamma_max_adopted = _gamma_max(delta_adopted)

    rows: List[Dict[str, Any]] = []
    for ex in examples:
        gamma_obs = float(ex.energy_ev) / float(ex.rest_mass_ev) if ex.rest_mass_ev != 0 else float("nan")
        delta_upper = _delta_upper_for_gamma(gamma_obs)
        ratio = (gamma_max_adopted / gamma_obs) if (math.isfinite(gamma_max_adopted) and gamma_obs > 0) else float("nan")
        rows.append(
            {
                "key": ex.key,
                "label": ex.label,
                "energy_ev": ex.energy_ev,
                "rest_mass_ev": ex.rest_mass_ev,
                "gamma_obs": gamma_obs,
                "log10_gamma_obs": math.log10(gamma_obs) if gamma_obs > 0 else float("nan"),
                "delta_upper_from_gamma": delta_upper,
                "log10_delta_upper": math.log10(delta_upper) if (math.isfinite(delta_upper) and delta_upper > 0) else float("nan"),
                "gamma_max_for_delta_adopted": gamma_max_adopted,
                "log10_gamma_max_for_delta_adopted": math.log10(gamma_max_adopted)
                if (math.isfinite(gamma_max_adopted) and gamma_max_adopted > 0)
                else float("nan"),
                "margin_gamma_max_over_gamma_obs": ratio,
                "notes": ex.notes,
                "source": ex.source,
            }
        )

    # Save CSV (no pandas dependency)

    csv_path = localize_figure_output_path(out_dir / "delta_saturation_constraints.csv", root=root, locale=FIGURE_LOCALE)
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in header))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Save JSON
    json_path = localize_figure_output_path(out_dir / "delta_saturation_constraints.json", root=root, locale=FIGURE_LOCALE)
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(in_path).replace("\\", "/"),
        "delta_adopted": delta_adopted,
        "gamma_max_for_delta_adopted": gamma_max_adopted,
        "model_notes": [
            "P-modelの速度項は (dτ/dt)_v = sqrt((1 - v^2/c^2 + δ)/(1+δ)) を採用する。",
            "このとき γ_max = lim(v→c) dt/dτ = sqrt((1+δ)/δ) ≈ 1/sqrt(δ) となり、SRのような発散は起きない。",
            "既存の高γ観測（加速器/宇宙線/ニュートリノ等）と矛盾しないためには δ が十分小さい必要がある。",
        ],
        "rows": rows,
        "outputs": {
            "csv": str(csv_path).replace("\\", "/"),
            "png": str(localize_figure_output_path(out_dir / "delta_saturation_constraints.png", root=root, locale=FIGURE_LOCALE)).replace("\\", "/"),
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot
    png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt

        _configure_font()

        labels_compact = [
            _t("LHC陽子\n(7 TeV)", "LHC proton\n(7 TeV)"),
            _t("超高エネルギー宇宙線\n(~10^20 eV)", "UHE cosmic ray\n(~10^20 eV)"),
            _t("高エネルギーν\n(10 PeV 仮定)", "High-energy ν\n(10 PeV)"),
        ]
        if len(labels_compact) != len(rows):
            labels_compact = [str(r["label"]) for r in rows]

        y = np.arange(len(labels_compact), dtype=float)
        profile_name = "part2_astrophysics"
        title_fs = get_wavep_font_size("title", name=profile_name)
        axis_fs = get_wavep_font_size("axis", name=profile_name)
        tick_fs = get_wavep_font_size("tick", name=profile_name)
        note_fs = get_wavep_font_size("note", name=profile_name)

        gamma_vals = [float(r["gamma_obs"]) for r in rows]
        delta_uppers = [float(r["delta_upper_from_gamma"]) for r in rows]

        log_gamma = [math.log10(g) if g > 0 else float("nan") for g in gamma_vals]
        log_delta_upper = [math.log10(d) if (math.isfinite(d) and d > 0) else float("nan") for d in delta_uppers]

        log_gamma_max = math.log10(gamma_max_adopted) if (math.isfinite(gamma_max_adopted) and gamma_max_adopted > 0) else float("nan")
        log_delta_adopted = math.log10(delta_adopted) if (math.isfinite(delta_adopted) and delta_adopted > 0) else float("nan")

        fig, (ax0, ax1) = plt.subplots(2, 1, sharey=True)
        apply_wavep_figure_layout(fig, template="part2_two_panel_tall")
        fig.subplots_adjust(left=0.205, right=0.982, top=0.925, bottom=0.115, hspace=0.42)

        ax0.barh(y, log_gamma, color="#1f77b4", alpha=0.92)
        if math.isfinite(log_gamma_max):
            ax0.axvline(
                log_gamma_max,
                color="#d62728",
                linestyle="--",
                linewidth=1.8,
            )

        ax0.set_yticks(y)
        ax0.set_yticklabels(labels_compact, fontsize=tick_fs)
        ax0.invert_yaxis()
        ax0.set_xlabel(_t("log10(γ_obs)", "log10(γ_obs)"), fontsize=axis_fs)
        ax0.set_title(_t("既存観測で到達しているローレンツ因子", "Observed Lorentz-factor scale"), fontsize=title_fs, pad=5.0)
        ax0.grid(True, axis="x", alpha=0.25)
        ax0.tick_params(axis="both", labelsize=tick_fs)

        for yi, g in enumerate(gamma_vals):
            ax0.text(log_gamma[yi] + 0.18, yi, f"γ≈{_fmt_sci(g, digits=1)}", ha="left", va="center", fontsize=note_fs)

        if math.isfinite(log_gamma_max):
            ax0.text(
                log_gamma_max - 0.10,
                -0.42,
                _t(f"γ_max（δ={_fmt_sci(delta_adopted)}）", f"γ_max (δ={_fmt_sci(delta_adopted)})"),
                ha="right",
                va="center",
                fontsize=note_fs,
                color="#d62728",
            )

        ax1.barh(y, log_delta_upper, color="#2ca02c", alpha=0.92)
        if math.isfinite(log_delta_adopted):
            ax1.axvline(
                log_delta_adopted,
                color="#d62728",
                linestyle="--",
                linewidth=1.8,
            )

        ax1.set_yticks(y)
        ax1.set_yticklabels(labels_compact, fontsize=tick_fs)
        ax1.set_xlabel(r"log10($\delta_{\mathrm{upper}}$)", fontsize=axis_fs)
        ax1.set_title(_t("既存観測からの δ 上限（概算）", "Approximate δ upper bounds"), fontsize=title_fs, pad=5.0)
        ax1.grid(True, axis="x", alpha=0.25)
        ax1.tick_params(axis="both", labelsize=tick_fs)

        for yi, d in enumerate(delta_uppers):
            ax1.text(log_delta_upper[yi] + 0.38, yi, _fmt_sci(d, digits=1), ha="left", va="center", fontsize=note_fs)

        if math.isfinite(log_gamma_max):
            ax0.set_xlim(0.0, max(log_gamma_max, max(v for v in log_gamma if math.isfinite(v))) + 1.2)

        if math.isfinite(log_delta_adopted):
            ax1.set_xlim(min(log_delta_adopted, min(v for v in log_delta_upper if math.isfinite(v))) - 3.4, 0.5)

        if math.isfinite(log_delta_adopted):
            ax1.text(
                log_delta_adopted + 0.28,
                -0.42,
                _t(f"採用δ={_fmt_sci(delta_adopted)}", f"adopted δ={_fmt_sci(delta_adopted)}"),
                ha="left",
                va="center",
                fontsize=note_fs,
                color="#d62728",
            )

        png_path = localize_figure_output_path(out_dir / "delta_saturation_constraints.png", root=root, locale=FIGURE_LOCALE)
        pdf_path = localize_figure_output_path(out_dir / "delta_saturation_constraints.pdf", root=root, locale=FIGURE_LOCALE)
        fig.savefig(png_path, dpi=220)
        fig.savefig(pdf_path)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] plot skipped: {e}")
        png_path = None

    mirror_dirs = [private_out_dir, public_out_dir]
    generated_paths = [csv_path, json_path]
    # 条件分岐: `isinstance(png_path, Path)` を満たす経路を評価する。
    if isinstance(png_path, Path):
        generated_paths.append(png_path)
        generated_paths.append(pdf_path)

    for dst_dir in mirror_dirs:
        for src in generated_paths:
            dst = localize_figure_output_path(dst_dir / src.name, root=root, locale=FIGURE_LOCALE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            # 条件分岐: `src.resolve() == dst.resolve()` を満たす経路を評価する。
            if src.resolve() == dst.resolve():
                continue

            shutil.copy2(src, dst)

    try:
        worklog.append_event(
            {
                "event_type": "theory_delta_saturation_constraints",
                "argv": sys.argv,
                "inputs": {"examples": in_path},
                "outputs": {
                    "csv": csv_path,
                    "json": json_path,
                    "png": (png_path if isinstance(png_path, Path) else None),
                    "pdf": (pdf_path if isinstance(png_path, Path) else None),
                },
                "metrics": {
                    "delta_adopted": delta_adopted,
                    "gamma_max": gamma_max_adopted,
                    "n_examples": len(rows),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] csv : {csv_path}")
    print(f"[ok] json: {json_path}")
    print(f"[ok] mirror(private): {private_out_dir}")
    print(f"[ok] mirror(public) : {public_out_dir}")
    # 条件分岐: `isinstance(png_path, Path)` を満たす経路を評価する。
    if isinstance(png_path, Path):
        print(f"[ok] png : {png_path}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
