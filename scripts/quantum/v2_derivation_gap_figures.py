from __future__ import annotations

"""
v2_derivation_gap_figures.py

既存公開 metrics から native α selector theorem の support surface と
weak-sector W/Z anchor checkpoint を、
公開済み metrics / rows から紙面向けの要約図へ再構成する。

出力:
  - output/private/quantum/v2_trial2_theorem_support_summary.pdf
  - output/private/quantum/v2_trial2_theorem_support_summary.png
  - output/public/quantum/v2_trial2_theorem_support_summary.pdf
  - output/public/quantum/v2_trial2_theorem_support_summary.png
  - output/private/quantum/v2_trial2_theorem_support_summary_metrics.json
  - output/public/quantum/v2_trial2_theorem_support_summary_metrics.json
  - output/private/quantum/v2_trial3_weak_checkpoint_summary.pdf
  - output/private/quantum/v2_trial3_weak_checkpoint_summary.png
  - output/public/quantum/v2_trial3_weak_checkpoint_summary.pdf
  - output/public/quantum/v2_trial3_weak_checkpoint_summary.png
  - output/private/quantum/v2_trial3_weak_checkpoint_summary_metrics.json
  - output/public/quantum/v2_trial3_weak_checkpoint_summary_metrics.json
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.plot_style import (  # noqa: E402
    apply_wavep_figure_layout,
    get_wavep_font_size,
    install_wavep_font_profile,
    resolve_wavep_cjk_font_family,
)


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl

        preferred = resolve_wavep_cjk_font_family(preferred_name="Noto Sans CJK JP")
        if preferred:
            mpl.rcParams["font.family"] = [preferred, "DejaVu Sans"]
            mpl.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]

        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_save_figure_bundle` の入出力契約と処理意図を定義する。

def _save_figure_bundle(fig: Figure, *, stem: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    targets = {
        "private": ROOT / "output" / "private" / "quantum",
        "public": ROOT / "output" / "public" / "quantum",
    }

    for scope, out_dir in targets.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"{stem}.pdf"
        png_path = out_dir / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
        fig.savefig(png_path, dpi=200, bbox_inches="tight", format="png")
        outputs[f"{scope}_pdf"] = _rel(pdf_path)
        outputs[f"{scope}_png"] = _rel(png_path)

    plt.close(fig)
    return outputs


# 関数: `_save_metrics_bundle` の入出力契約と処理意図を定義する。

def _save_metrics_bundle(*, stem: str, payload: dict[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    targets = {
        "private": ROOT / "output" / "private" / "quantum",
        "public": ROOT / "output" / "public" / "quantum",
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for scope, out_dir in targets.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{stem}_metrics.json"
        json_path.write_text(text, encoding="utf-8")
        outputs[f"{scope}_json"] = _rel(json_path)

    return outputs


# 関数: `_build_trial2_support_bundle` の入出力契約と処理意図を定義する。

def _build_trial2_support_bundle() -> tuple[Figure, dict[str, Any]]:
    sign_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "q_8_7_56_1963_1966_asymp_sign_parity_audit_declaration_gate_metrics.json"
    )
    coherence_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "q_8_7_56_1783_1786_int_coh_hh_reactivation_declaration_gate_metrics.json"
    )
    hh_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "q_8_7_56_1791_1794_hh_surface_reactivation_declaration_gate_metrics.json"
    )
    lattice_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "q_8_7_56_2055_2058_higher_harmonic_lattice_loading_declaration_gate_metrics.json"
    )

    roots = np.asarray(sign_metrics["evidence"]["high_q_signed_zero_roots_over_m0"], dtype=float)
    r_box = float(sign_metrics["summary"]["solver_box_edge_over_m0"])
    half_integer_indices = np.round(roots * r_box / math.pi - 0.5).astype(int)
    theory_roots = (half_integer_indices + 0.5) * math.pi / r_box
    root_index = np.arange(1, roots.size + 1, dtype=float)

    ff_amplitude = float(coherence_metrics["inputs"]["constants"]["field_strength_response_at_q_theory"])
    hh_amplitude = float(hh_metrics["summary"]["exact_hh_amplitude_at_q_theory"])
    fh_amplitude = float(hh_metrics["summary"]["exact_fh_amplitude_at_q_theory"])
    lambda_plus = float(hh_metrics["summary"]["exact_lambda_plus_at_q_theory"])

    mismatch_labels = ["3-8", "9-16", "17-24"]
    theorem_mismatch = np.asarray(
        [
            lattice_metrics["summary"]["theorem_fit_window_max_mismatch_fraction"],
            lattice_metrics["summary"]["theorem_extension_window_max_mismatch_fraction"],
            lattice_metrics["summary"]["theorem_farther_window_max_mismatch_fraction"],
        ],
        dtype=float,
    )
    searched_mismatch = np.asarray(
        [
            lattice_metrics["summary"]["searched_fit_window_max_mismatch_fraction"],
            lattice_metrics["summary"]["searched_extension_window_max_mismatch_fraction"],
            lattice_metrics["summary"]["searched_farther_window_max_mismatch_fraction"],
        ],
        dtype=float,
    )

    alpha_labels = ["スカラー", "ランク1代理", "厳密混合"]
    alpha_values = np.asarray(
        [
            coherence_metrics["inputs"]["constants"]["scalar_alpha_exact_at_q_theory"],
            coherence_metrics["summary"]["rank_one_alpha_with_energy_proxy"],
            hh_metrics["summary"]["exact_alpha_mix_at_q_theory"],
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, dpi=150)
    apply_wavep_figure_layout(fig, template="part2_quad_panel_spacious")
    title_font = get_wavep_font_size("title")
    axis_font = get_wavep_font_size("axis")
    tick_font = get_wavep_font_size("tick")
    legend_font = get_wavep_font_size("legend")
    note_font = get_wavep_font_size("note")
    suptitle_font = get_wavep_font_size("suptitle") + 1.8

    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.plot(root_index, roots, color="#0f766e", marker="o", ms=2.8, lw=1.1, label="観測零点")
    ax0.plot(root_index, theory_roots, color="#b45309", lw=1.0, ls="--", label=r"理論格子 $(n+1/2)\pi/R_{\rm box}$")
    ax0.set_title("符号交代: 高 q 零点格子", fontsize=title_font)
    ax0.set_xlabel("零点番号", fontsize=axis_font)
    ax0.set_ylabel(r"$q/m_0$", fontsize=axis_font)
    ax0.tick_params(labelsize=tick_font)
    ax0.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax0.legend(frameon=False, fontsize=legend_font, loc="upper left")
    ax0.text(
        0.03,
        0.03,
        (
            f"0<=q/m0<=4 を厳密保持\n"
            f"平均間隔差 = {sign_metrics['summary']['spacing_rel_gap_vs_theory']:.2e}\n"
            f"最大格子誤差 = {sign_metrics['summary']['root_lattice_max_abs_error']:.4f}"
        ),
        transform=ax0.transAxes,
        fontsize=note_font,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    amp_labels = ["FF", "FH", "HH", "lambda+"]
    amp_values = np.asarray([ff_amplitude, fh_amplitude, hh_amplitude, lambda_plus], dtype=float)
    amp_colors = ["#1d4ed8", "#7c3aed", "#ea580c", "#059669"]
    ax1.bar(amp_labels, amp_values, color=amp_colors, alpha=0.9)
    ax1.set_title(r"$q_{\mathrm{theory}}$ におけるランク1閉包", fontsize=title_font)
    ax1.set_ylabel("振幅", fontsize=axis_font)
    ax1.tick_params(labelsize=tick_font)
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax1.text(
        0.03,
        0.03,
        (
            "ρ_exact = 1\n"
            f"HH 閾値 = {coherence_metrics['summary']['rank_one_hh_threshold_for_scalar']:.6f}\n"
            f"α_mix = {hh_metrics['summary']['exact_alpha_mix_at_q_theory']:.9f}"
        ),
        transform=ax1.transAxes,
        fontsize=note_font,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    x = np.arange(len(mismatch_labels), dtype=float)
    width = 0.34
    ax2.bar(x - width / 2.0, theorem_mismatch * 100.0, width=width, color="#0f766e", label="定理基準")
    ax2.bar(x + width / 2.0, searched_mismatch * 100.0, width=width, color="#9a3412", label="探索基準")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mismatch_labels, fontsize=tick_font)
    ax2.set_title("高調波格子の区分的延長窓", fontsize=title_font)
    ax2.set_xlabel("高調波窓", fontsize=axis_font)
    ax2.set_ylabel("最大不一致 [%]", fontsize=axis_font)
    ax2.tick_params(labelsize=tick_font)
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax2.legend(frameon=False, fontsize=legend_font, loc="upper right")
    ax2.text(
        0.03,
        0.03,
        (
            f"格子刻み = {lattice_metrics['summary']['bulk_delta_r_over_m0']:.3f}\n"
            f"定理基準 = {lattice_metrics['summary']['theorem_lattice_base_over_m0']:.4f}\n"
            f"探索基準との差 = {lattice_metrics['summary']['theorem_vs_searched_base_gap_over_m0']:.4f}"
        ),
        transform=ax2.transAxes,
        fontsize=note_font,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    ax3.bar(alpha_labels, alpha_values * 1000.0, color=["#0369a1", "#c2410c", "#16a34a"], alpha=0.9)
    ax3.set_title("固定 q における α 支持", fontsize=title_font)
    ax3.set_ylabel(r"$10^3 \times \alpha$", fontsize=axis_font)
    ax3.tick_params(labelsize=tick_font)
    ax3.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax3.text(
        0.03,
        0.03,
        (
            f"代理量/厳密 HH 比 = {hh_metrics['summary']['proxy_to_exact_hh_ratio']:.3f}\n"
            f"局所閉包 = {hh_metrics['summary']['branch_local_completion_only']}\n"
            "全 q HH 面は未完"
        ),
        transform=ax3.transAxes,
        fontsize=note_font,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    fig.suptitle("微細構造定数の選別定理の支持面", fontsize=suptitle_font, y=0.985)

    metrics = {
        "generated_utc": _iso_utc_now(),
        "figure_stem": "v2_trial2_theorem_support_summary",
        "sources": {
            "sign_parity": "output/public/quantum/q_8_7_56_1963_1966_asymp_sign_parity_audit_declaration_gate_metrics.json",
            "rank_one_coherence": "output/public/quantum/q_8_7_56_1783_1786_int_coh_hh_reactivation_declaration_gate_metrics.json",
            "fixed_q_completion": "output/public/quantum/q_8_7_56_1791_1794_hh_surface_reactivation_declaration_gate_metrics.json",
            "bulk_lattice": "output/public/quantum/q_8_7_56_2055_2058_higher_harmonic_lattice_loading_declaration_gate_metrics.json",
        },
        "summary": {
            "retained_interval_over_m0": sign_metrics["summary"]["retained_interval_over_m0"],
            "asymptotic_audit_interval_over_m0": sign_metrics["summary"]["asymptotic_audit_interval_over_m0"],
            "mean_high_q_spacing": sign_metrics["summary"]["mean_high_q_spacing"],
            "spacing_rel_gap_vs_theory": sign_metrics["summary"]["spacing_rel_gap_vs_theory"],
            "root_lattice_max_abs_error": sign_metrics["summary"]["root_lattice_max_abs_error"],
            "exact_internal_rank_one_coherence_derived": coherence_metrics["summary"]["exact_internal_rank_one_coherence_derived"],
            "rank_one_hh_threshold_for_scalar": coherence_metrics["summary"]["rank_one_hh_threshold_for_scalar"],
            "exact_hh_amplitude_at_q_theory": hh_metrics["summary"]["exact_hh_amplitude_at_q_theory"],
            "exact_fh_amplitude_at_q_theory": hh_metrics["summary"]["exact_fh_amplitude_at_q_theory"],
            "exact_lambda_plus_at_q_theory": hh_metrics["summary"]["exact_lambda_plus_at_q_theory"],
            "exact_alpha_mix_at_q_theory": hh_metrics["summary"]["exact_alpha_mix_at_q_theory"],
            "theorem_fit_window_max_mismatch_fraction": lattice_metrics["summary"]["theorem_fit_window_max_mismatch_fraction"],
            "theorem_extension_window_max_mismatch_fraction": lattice_metrics["summary"]["theorem_extension_window_max_mismatch_fraction"],
            "theorem_farther_window_max_mismatch_fraction": lattice_metrics["summary"]["theorem_farther_window_max_mismatch_fraction"],
            "exact_loading_index_theorem_available": lattice_metrics["summary"]["exact_loading_index_theorem_available"],
        },
    }
    return fig, metrics


# 関数: `_build_trial3_checkpoint_bundle` の入出力契約と処理意図を定義する。

def _build_trial3_checkpoint_bundle() -> tuple[Figure, dict[str, Any]]:
    closeout_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json"
    )
    solver_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "mass_origin_v2_trial3_two_component_shooting_solver_implementation_metrics.json"
    )
    ode_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_metrics.json"
    )
    spectrum_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json"
    )
    charge_metrics = _read_json(
        ROOT / "output" / "public" / "quantum" / "mass_origin_v2_t3_t2_charge_window_pivot_source_inventory_metrics.json"
    )

    error_labels = ["W", "Z"]
    error_percent = np.asarray(
        [
            closeout_metrics["summary"]["exact_w_relative_error"] * 100.0,
            closeout_metrics["summary"]["exact_z_relative_error"] * 100.0,
        ],
        dtype=float,
    )
    kappa_values = np.asarray(
        [
            closeout_metrics["summary"]["exact_w_kappa_coupled"],
            closeout_metrics["summary"]["exact_z_kappa_coupled"],
        ],
        dtype=float,
    )
    count_labels = ["粗探索", "局在解", "整数モード", "厳密行数"]
    count_values = np.asarray(
        [
            solver_metrics["summary"]["two_component_smoke_scan_size"],
            spectrum_metrics["summary"]["localized_solution_count_total"],
            spectrum_metrics["summary"]["base_mode_count_total"],
            spectrum_metrics["summary"]["exact_vector_row_count_total"],
        ],
        dtype=float,
    )
    current_window = charge_metrics["summary"]["current_charge_window_or_none"]
    extended_window = charge_metrics["summary"]["extended_charge_window_or_none"]

    fig, axes = plt.subplots(2, 2, dpi=150)
    apply_wavep_figure_layout(fig, template="part2_quad_panel_spacious")
    title_font = get_wavep_font_size("title")
    axis_font = get_wavep_font_size("axis")
    tick_font = get_wavep_font_size("tick")
    note_font = get_wavep_font_size("note")
    suptitle_font = get_wavep_font_size("suptitle") + 1.8

    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.bar(error_labels, error_percent, color=["#2563eb", "#16a34a"], alpha=0.9)
    ax0.set_title("同一系列上の質量基準点の誤差", fontsize=title_font)
    ax0.set_ylabel("相対誤差 [%]", fontsize=axis_font)
    ax0.yaxis.set_major_formatter(PercentFormatter())
    ax0.tick_params(labelsize=tick_font)
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax0.text(
        0.03,
        0.03,
        (
            f"family = ({closeout_metrics['summary']['anchor_family_or_none']['k']}, "
            f"{closeout_metrics['summary']['anchor_family_or_none']['ell']}, "
            f"{closeout_metrics['summary']['anchor_family_or_none']['s']})\n"
            "closeout 規模: 約 0.002%"
        ),
        transform=ax0.transAxes,
        fontsize=note_font,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    ax1.bar(error_labels, kappa_values, color=["#0f766e", "#b45309"], alpha=0.9)
    ax1.set_title("結合局在", fontsize=title_font)
    ax1.set_ylabel(r"$\kappa_{\rm coupled}$", fontsize=axis_font)
    ax1.tick_params(labelsize=tick_font)
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax1.text(
        0.03,
        0.03,
        (
            "単成分 clip は再分類\n"
            "物理局在は結合固有モードで判定"
        ),
        transform=ax1.transAxes,
        fontsize=note_font,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    ax2.bar(count_labels, count_values, color=["#7c3aed", "#0891b2", "#ea580c", "#475569"], alpha=0.9)
    ax2.set_title("数値解法とスペクトルの尺度", fontsize=title_font)
    ax2.set_ylabel("件数（対数軸）", fontsize=axis_font)
    ax2.set_yscale("log")
    ax2.tick_params(labelsize=tick_font)
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax2.text(
        0.03,
        0.03,
        (
            f"状態ベクトル = {int(solver_metrics['rows'][1]['value'])} 状態\n"
            f"局在した ell 集合 = {spectrum_metrics['summary']['localized_ell_values']}\n"
            f"新規パラメータ数 = {ode_metrics['summary']['new_free_parameters_introduced']}"
        ),
        transform=ax2.transAxes,
        fontsize=note_font,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    ax3.hlines([1.0, 0.0], [current_window[0], extended_window[0]], [current_window[1], extended_window[1]], colors=["#2563eb", "#16a34a"], lw=4.0)
    ax3.scatter(current_window, [1.0, 1.0], color="#2563eb", s=24)
    ax3.scatter(extended_window, [0.0, 0.0], color="#16a34a", s=24)
    ax3.set_title("理論基準で使う電荷窓拡張", fontsize=title_font)
    ax3.set_xlabel("電荷代理 n", fontsize=axis_font)
    ax3.set_yticks([0.0, 1.0])
    ax3.set_yticklabels(["拡張", "現行"], fontsize=tick_font)
    ax3.tick_params(labelsize=tick_font)
    ax3.grid(True, axis="x", ls=":", lw=0.6, alpha=0.7)
    ax3.text(
        0.03,
        0.03,
        (
            f"現行 = [{current_window[0]}, {current_window[1]}]\n"
            f"拡張 = [{extended_window[0]}, {extended_window[1]}]\n"
            "拡張後も同一系列の支持を保持"
        ),
        transform=ax3.transAxes,
        fontsize=note_font,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    fig.suptitle("W/Z 質量基準点の結合局在条件による理論基準", fontsize=suptitle_font, y=0.985)

    metrics = {
        "generated_utc": _iso_utc_now(),
        "figure_stem": "v2_trial3_weak_checkpoint_summary",
        "sources": {
            "coupled_localization_closeout": "output/public/quantum/mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json",
            "shooting_solver": "output/public/quantum/mass_origin_v2_trial3_two_component_shooting_solver_implementation_metrics.json",
            "coupled_radial_ode": "output/public/quantum/mass_origin_v2_trial3_two_component_coupled_radial_ode_derivation_metrics.json",
            "spectrum_scan": "output/public/quantum/mass_origin_v2_trial3_two_component_spectrum_computation_metrics.json",
            "charge_window_pivot": "output/public/quantum/mass_origin_v2_t3_t2_charge_window_pivot_source_inventory_metrics.json",
        },
        "summary": {
            "anchor_family_or_none": closeout_metrics["summary"]["anchor_family_or_none"],
            "exact_w_relative_error": closeout_metrics["summary"]["exact_w_relative_error"],
            "exact_z_relative_error": closeout_metrics["summary"]["exact_z_relative_error"],
            "exact_w_kappa_coupled": closeout_metrics["summary"]["exact_w_kappa_coupled"],
            "exact_z_kappa_coupled": closeout_metrics["summary"]["exact_z_kappa_coupled"],
            "two_component_shooting_solver_implemented": solver_metrics["summary"]["two_component_shooting_solver_implemented"],
            "localized_solution_count_total": spectrum_metrics["summary"]["localized_solution_count_total"],
            "exact_vector_row_count_total": spectrum_metrics["summary"]["exact_vector_row_count_total"],
            "current_charge_window_or_none": charge_metrics["summary"]["current_charge_window_or_none"],
            "extended_charge_window_or_none": charge_metrics["summary"]["extended_charge_window_or_none"],
        },
    }
    return fig, metrics


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    install_wavep_font_profile(profile_name="part3a_quantum_foundations")
    _set_japanese_font()

    trial2_fig, trial2_metrics = _build_trial2_support_bundle()
    trial2_outputs = _save_figure_bundle(trial2_fig, stem="v2_trial2_theorem_support_summary")
    trial2_outputs.update(_save_metrics_bundle(stem="v2_trial2_theorem_support_summary", payload=trial2_metrics))
    print(f"[ok] trial2 outputs: {json.dumps(trial2_outputs, ensure_ascii=False)}")

    trial3_fig, trial3_metrics = _build_trial3_checkpoint_bundle()
    trial3_outputs = _save_figure_bundle(trial3_fig, stem="v2_trial3_weak_checkpoint_summary")
    trial3_outputs.update(_save_metrics_bundle(stem="v2_trial3_weak_checkpoint_summary", payload=trial3_metrics))
    print(f"[ok] trial3 outputs: {json.dumps(trial3_outputs, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
