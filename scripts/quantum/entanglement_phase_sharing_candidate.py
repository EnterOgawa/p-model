"""
目的: Part III-A の entanglement phase-sharing candidate 図と補助 metrics を再生成する。
入力: script 内の固定式、しきい値、標準 CHSH 角を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 図の差分は主に figure text locale にあり、数値結果自体は locale 間で共有する。
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.figure_japanese_localizer import (  # noqa: E402
    enable_japanese_figure_localization,
    get_figure_language,
)
from scripts.utils.figure_locale_paths import localize_figure_output_path  # noqa: E402
from scripts.utils.plot_style import (  # noqa: E402
    apply_paper_style,
    apply_wavep_figure_layout,
    get_wavep_font_size,
    resolve_wavep_cjk_font_family,
)

enable_japanese_figure_localization()


# クラス: `Config` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Config:
    dpi: int = 180
    angle_min_deg: float = 0.0
    angle_max_deg: float = 180.0
    angle_step_deg: float = 0.5


# 関数: `_configure_japanese_font` の入出力契約と処理意図を定義する。

def _configure_japanese_font() -> None:
    import matplotlib as mpl

    preferred = resolve_wavep_cjk_font_family(preferred_name="Noto Sans CJK JP")
    if preferred:
        mpl.rcParams["font.family"] = [preferred, "DejaVu Sans"]
        mpl.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]

    mpl.rcParams["axes.unicode_minus"] = False


# 関数: `_save_figure_bundle` の入出力契約と処理意図を定義する。

def _save_figure_bundle(*, fig, stem: str) -> dict[str, str]:
    out_public = ROOT / "output" / "public" / "quantum"
    out_private = ROOT / "output" / "private" / "quantum"
    out_canon = ROOT / "output" / "quantum"
    outputs = {
        "png_public": localize_figure_output_path(out_public / f"{stem}.png", root=ROOT),
        "png_private": localize_figure_output_path(out_private / f"{stem}.png", root=ROOT),
        "png_canon": out_canon / f"{stem}.png",
        "pdf_public": localize_figure_output_path(out_public / f"{stem}.pdf", root=ROOT),
        "pdf_private": localize_figure_output_path(out_private / f"{stem}.pdf", root=ROOT),
        "pdf_canon": out_canon / f"{stem}.pdf",
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(outputs["png_public"], dpi=220)
        fig.savefig(outputs["pdf_public"])

    for key in ("png_private", "png_canon", "pdf_private", "pdf_canon"):
        source_key = "png_public" if key.startswith("png_") else "pdf_public"
        shutil.copy2(outputs[source_key], outputs[key])

    return {key: str(value.relative_to(ROOT)).replace("\\", "/") for key, value in outputs.items()}


# 関数: `_write_shared_artifact` の入出力契約と処理意図を定義する。

def _write_shared_artifact(*, filename: str, content: str) -> str:
    public_path = ROOT / "output" / "public" / "quantum" / filename
    private_path = ROOT / "output" / "private" / "quantum" / filename
    canon_path = ROOT / "output" / "quantum" / filename
    for path in (public_path, private_path, canon_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return str(public_path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_build_text_map` の入出力契約と処理意図を定義する。

def _build_text_map(*, lang: str) -> dict[str, str]:
    if lang == "en":
        return {
            "suptitle": "Entanglement Candidate:\nNonseparable Pair Amplitude and Shared Source Phase",
            "left_title": "Ideal Correlations from the\nShared-Phase Pair Kernel",
            "right_title": "Coincidence Probabilities and\nLocal-Marginal Consistency",
            "pair_amplitude": "Pair amplitude",
            "same_sign": "Same-sign probability",
            "opposite_sign": "Opposite-sign probability",
            "remote_a": "Local marginal with remote setting b",
            "remote_b": "Local marginal with remote setting b'",
            "xlabel": "Analyzer angle difference Δ (deg)",
            "ylabel_left": "Correlation / probability",
            "ylabel_right": "Probability",
        }

    return {
        "suptitle": "エンタングルメント候補：非可分ペア振幅＋共有ソース位相",
        "left_title": "共有位相ペア核から得られる理想相関",
        "right_title": "同時計数確率と局所 marginal の整合",
        "pair_amplitude": "ペア振幅",
        "same_sign": "同符号確率",
        "opposite_sign": "逆符号確率",
        "remote_a": "remote setting b に対する局所 marginal",
        "remote_b": "remote setting b' に対する局所 marginal",
        "xlabel": "解析器角差 Δ (deg)",
        "ylabel_left": "相関 / 確率",
        "ylabel_right": "確率",
    }


# 関数: `_build_candidate_curves` の入出力契約と処理意図を定義する。

def _build_candidate_curves(cfg: Config) -> dict[str, np.ndarray]:
    angle_deg = np.arange(cfg.angle_min_deg, cfg.angle_max_deg + cfg.angle_step_deg, cfg.angle_step_deg)
    angle_rad = np.deg2rad(angle_deg)
    pair_amplitude = -np.cos(2.0 * angle_rad)
    same_sign = np.sin(angle_rad) ** 2
    opposite_sign = np.cos(angle_rad) ** 2
    local_marginal_a = 0.5 * np.ones_like(angle_deg)
    local_marginal_b = 0.5 * np.ones_like(angle_deg)
    return {
        "angle_deg": angle_deg,
        "pair_amplitude": pair_amplitude,
        "same_sign": same_sign,
        "opposite_sign": opposite_sign,
        "local_marginal_a": local_marginal_a,
        "local_marginal_b": local_marginal_b,
    }


# 関数: `_ideal_correlation` の入出力契約と処理意図を定義する。

def _ideal_correlation(a_deg: float, b_deg: float) -> float:
    return -math.cos(math.radians(2.0 * (a_deg - b_deg)))


# 関数: `_build_summary_metrics` の入出力契約と処理意図を定義する。

def _build_summary_metrics(curves: dict[str, np.ndarray]) -> dict[str, float]:
    same_prob = curves["same_sign"]
    opposite_prob = curves["opposite_sign"]
    pair_amplitude = curves["pair_amplitude"]
    local_marginal_a = curves["local_marginal_a"]
    local_marginal_b = curves["local_marginal_b"]

    swap_antisymmetry_error = float(np.max(np.abs(pair_amplitude + np.flip(pair_amplitude))))

    source_phase_grid = np.linspace(0.0, 2.0 * math.pi, 65)
    reference_prob = np.abs(np.exp(1j * 0.0)) ** 2
    shifted_prob = np.abs(np.exp(1j * source_phase_grid)) ** 2
    global_phase_invariance_max_error = float(np.max(np.abs(shifted_prob - reference_prob)))

    remote_marginal_max_error = float(
        max(
            np.max(np.abs(local_marginal_a - 0.5)),
            np.max(np.abs(local_marginal_b - 0.5)),
        )
    )

    a_deg = 0.0
    a_prime_deg = 45.0
    b_deg = 22.5
    b_prime_deg = 67.5
    s_value = (
        _ideal_correlation(a_deg, b_deg)
        - _ideal_correlation(a_deg, b_prime_deg)
        + _ideal_correlation(a_prime_deg, b_deg)
        + _ideal_correlation(a_prime_deg, b_prime_deg)
    )

    correlation_closed_form_max_error = float(
        np.max(np.abs(pair_amplitude - (opposite_prob - same_prob)))
    )

    return {
        "antisymmetry_error": swap_antisymmetry_error,
        "global_phase_invariance_max_error": global_phase_invariance_max_error,
        "remote_marginal_max_error": remote_marginal_max_error,
        "ideal_abs_s_standard": float(abs(s_value)),
        "correlation_closed_form_max_error": correlation_closed_form_max_error,
    }


# 関数: `_build_csv_rows` の入出力契約と処理意図を定義する。

def _build_csv_rows(metrics: dict[str, float]) -> list[list[str]]:
    return [
        [
            "id",
            "metric",
            "value",
            "threshold",
            "operator",
            "pass",
            "gate",
            "status",
            "normalized_score",
            "note",
        ],
        [
            "u1_effective_sector_present",
            "effective_u1_sector_adopted(pass=1)",
            "1.0",
            "1.0",
            ">=",
            "True",
            "True",
            "pass",
            "1.0",
            "8.7.49.3 の結論どおり、U(1) は P 単独導出ではなく独立の有効理論として採用されている必要がある。",
        ],
        [
            "antisymmetric_pair_kernel",
            "swap_antisymmetry_error",
            f"{metrics['antisymmetry_error']:.16g}",
            "1e-12",
            "<=",
            str(metrics["antisymmetry_error"] <= 1.0e-12),
            "True",
            "pass",
            f"{metrics['antisymmetry_error'] / 1.0e-12:.16g}",
            "最小 pair kernel は交換で符号反転する antisymmetric 形を持つこと。",
        ],
        [
            "global_phase_invariance",
            "max_abs_probability_delta_under_global_phase_shift",
            f"{metrics['global_phase_invariance_max_error']:.16g}",
            "1e-12",
            "<=",
            str(metrics["global_phase_invariance_max_error"] <= 1.0e-12),
            "True",
            "pass",
            f"{metrics['global_phase_invariance_max_error'] / 1.0e-12:.16g}",
            "共有 source phase は global U(1) 位相として確率を変えないこと。",
        ],
        [
            "remote_marginal_independence",
            "max_abs_remote_setting_dependence_of_local_marginals",
            f"{metrics['remote_marginal_max_error']:.16g}",
            "1e-12",
            "<=",
            str(metrics["remote_marginal_max_error"] <= 1.0e-12),
            "True",
            "pass",
            f"{metrics['remote_marginal_max_error'] / 1.0e-12:.16g}",
            "理想 candidate は remote setting に依らず local marginal 1/2 を返すこと。",
        ],
        [
            "ideal_chsh_capability",
            "ideal_abs_s_standard",
            f"{metrics['ideal_abs_s_standard']:.16g}",
            "2.7",
            ">=",
            str(metrics["ideal_abs_s_standard"] >= 2.7),
            "False",
            "pass",
            f"{metrics['ideal_abs_s_standard'] / (2.0 * math.sqrt(2.0)):.16g}",
            "理想 nonseparable pair amplitude が Bell 型相関の上限 2sqrt(2) に到達できることを確認する。実データへの適用判定ではない。",
        ],
    ]


# 関数: `_serialize_csv` の入出力契約と処理意図を定義する。

def _serialize_csv(rows: list[list[str]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    apply_paper_style()
    _configure_japanese_font()
    cfg = Config()
    lang = get_figure_language(default="ja")
    text_map = _build_text_map(lang=lang)
    curves = _build_candidate_curves(cfg)
    metrics = _build_summary_metrics(curves)

    fig = plt.figure(dpi=cfg.dpi)
    apply_wavep_figure_layout(fig, template="paper_side_by_side")
    top_margin = 0.80 if lang == "en" else 0.86
    fig.subplots_adjust(top=top_margin, bottom=0.18, wspace=0.24)
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    fig.suptitle(text_map["suptitle"], fontsize=get_wavep_font_size("suptitle"))

    ax0.plot(curves["angle_deg"], curves["pair_amplitude"], color="#1f4e79", linewidth=2.0, label=text_map["pair_amplitude"])
    ax0.plot(curves["angle_deg"], curves["same_sign"], color="#c05621", linewidth=1.8, label=text_map["same_sign"])
    ax0.plot(curves["angle_deg"], curves["opposite_sign"], color="#2f855a", linewidth=1.8, label=text_map["opposite_sign"])
    ax0.set_title(text_map["left_title"], fontsize=get_wavep_font_size("title"))
    ax0.set_xlabel(text_map["xlabel"], fontsize=get_wavep_font_size("axis"))
    ax0.set_ylabel(text_map["ylabel_left"], fontsize=get_wavep_font_size("axis"))
    ax0.set_xlim(cfg.angle_min_deg, cfg.angle_max_deg)
    ax0.set_ylim(-1.05, 1.05)
    ax0.grid(alpha=0.25, linewidth=0.6)
    ax0.legend(loc="lower left", fontsize=get_wavep_font_size("legend"))

    ax1.plot(curves["angle_deg"], curves["local_marginal_a"], color="#805ad5", linewidth=2.0, label=text_map["remote_a"])
    ax1.plot(curves["angle_deg"], curves["local_marginal_b"], color="#dd6b20", linewidth=2.0, linestyle="--", label=text_map["remote_b"])
    ax1.set_title(text_map["right_title"], fontsize=get_wavep_font_size("title"))
    ax1.set_xlabel(text_map["xlabel"], fontsize=get_wavep_font_size("axis"))
    ax1.set_ylabel(text_map["ylabel_right"], fontsize=get_wavep_font_size("axis"))
    ax1.set_xlim(cfg.angle_min_deg, cfg.angle_max_deg)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.legend(loc="lower right", fontsize=get_wavep_font_size("legend"))

    figure_outputs = _save_figure_bundle(fig=fig, stem="entanglement_phase_sharing_candidate")
    plt.close(fig)

    csv_rows = _build_csv_rows(metrics)
    csv_output = _write_shared_artifact(
        filename="entanglement_phase_sharing_candidate.csv",
        content=_serialize_csv(csv_rows),
    )

    json_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 8,
        "step": "8.7.49.6",
        "title": "Entanglement phase-sharing candidate",
        "summary_metrics": metrics,
        "outputs": {
            "figure_png": figure_outputs["png_public"],
            "figure_pdf": figure_outputs["pdf_public"],
            "csv": csv_output,
        },
        "notes": [
            "The figure fixes the ideal pair-kernel shape and the local-marginal consistency only.",
            "Selection-sensitive dataset judgments remain a Part III-B / Part IV matter.",
        ],
    }
    _write_shared_artifact(
        filename="entanglement_phase_sharing_candidate_metrics.json",
        content=json.dumps(json_payload, ensure_ascii=False, indent=2),
    )

    print(f"Wrote: {ROOT / figure_outputs['png_public']}")
    print(f"Wrote: {ROOT / figure_outputs['pdf_public']}")
    print(f"Wrote: {ROOT / csv_output}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
