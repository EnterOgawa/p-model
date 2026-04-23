#!/usr/bin/env python3
"""
Evaluate the helper-backed selected-extension extra-q source-materialization surface.

This backend does not derive a new selected-extension-specific rescue. It replays
the actual helper-backed augmented-q surface

    Q_aug^(pilot-HS) = Q_ret union Q_ext^(ind)

and diagnoses whether the materialized extra-q checkpoints constitute a new
canonical rescue or merely carry over legacy Phase-3 blind sidebands.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.selected_extension_solver_side_extra_q_range_backend import (
    build_selected_extension_solver_side_extra_q_range_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
SCALAR_ALPHA_Q_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5375-5378",
        "updated_pack_scalar_proxy_alpha_q_curve_diagnosis_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]


# 関数: 必須 input artifact の存在を確認する。
def require(path: Path) -> None:
    """Abort immediately when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON artifact into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: blind form factor から scalar alpha proxy を計算する。

def compute_alpha_from_form_factor(form_factor_value: float) -> float:
    """Return alpha = F(q)^2 / (4 pi) for one form-factor value."""
    return float(form_factor_value * form_factor_value / (4.0 * math.pi))


# 関数: helper-backed extra-q numeric rerun diagnostic pack を構成する。

def build_selected_extension_solver_side_extra_q_range_numeric_rerun_pack(
    ell_values: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Evaluate the helper-backed augmented-q surface against scalar exact anchors."""
    require(SCALAR_ALPHA_Q_AUDIT)

    scalar_summary = read_json(SCALAR_ALPHA_Q_AUDIT)["summary"]
    qext_pack = build_selected_extension_solver_side_extra_q_range_pack(
        ell_values=ell_values
    )

    alpha_exact_at_q_theory = float(qext_pack["alpha_exact_at_q_theory"])
    alpha_target = float(scalar_summary["alpha_target"])
    scalar_q_exact_over_m0 = float(scalar_summary["primary_q_exact_over_m0"])
    q_theory_over_m0 = float(qext_pack["retained_q_window"]["q_theory_over_m0"])

    qext_labels = tuple(qext_pack["q_ext_ind_window"].keys())
    alpha_qext_pack: dict[str, float] = {}
    q_surface_diagnostics: dict[str, dict[str, float | bool | str]] = {}
    best_extra_label_vs_alpha_exact = None
    best_extra_label_vs_alpha_target = None
    best_extra_label_vs_q_exact = None
    best_extra_alpha_exact_residual = math.inf
    best_extra_alpha_target_residual = math.inf
    best_extra_q_exact_gap = math.inf

    for label, q_value in qext_pack["q_aug_window"].items():
        form_factor_value = float(qext_pack["F_blind_qext_pack"][label])
        alpha_value = compute_alpha_from_form_factor(form_factor_value)
        alpha_qext_pack[label] = alpha_value

        rel_vs_alpha_exact_q_theory = abs(alpha_value - alpha_exact_at_q_theory) / alpha_exact_at_q_theory
        rel_vs_alpha_target = abs(alpha_value - alpha_target) / alpha_target
        rel_q_gap_vs_scalar_q_exact = abs(float(q_value) - scalar_q_exact_over_m0) / scalar_q_exact_over_m0
        legacy_phase3_sideband = bool(label in qext_labels)
        source_phase = (
            "phase3_blind_numeric_evaluation"
            if legacy_phase3_sideband
            else "selected_extension_retained_surface"
        )

        q_surface_diagnostics[label] = {
            "q_over_m0": float(q_value),
            "F_blind_value": form_factor_value,
            "alpha_blind_value": alpha_value,
            "rel_vs_alpha_exact_q_theory": rel_vs_alpha_exact_q_theory,
            "rel_vs_alpha_target": rel_vs_alpha_target,
            "rel_q_gap_vs_scalar_q_exact": rel_q_gap_vs_scalar_q_exact,
            "legacy_phase3_sideband": legacy_phase3_sideband,
            "source_phase": source_phase,
        }

        if not legacy_phase3_sideband:
            continue

        if rel_vs_alpha_exact_q_theory < best_extra_alpha_exact_residual:
            best_extra_alpha_exact_residual = rel_vs_alpha_exact_q_theory
            best_extra_label_vs_alpha_exact = label

        if rel_vs_alpha_target < best_extra_alpha_target_residual:
            best_extra_alpha_target_residual = rel_vs_alpha_target
            best_extra_label_vs_alpha_target = label

        if rel_q_gap_vs_scalar_q_exact < best_extra_q_exact_gap:
            best_extra_q_exact_gap = rel_q_gap_vs_scalar_q_exact
            best_extra_label_vs_q_exact = label

    q_theory_diagnostic = q_surface_diagnostics["q_theory_over_m0"]
    q_theory_failure_preserved_now = bool(
        float(q_theory_diagnostic["F_blind_value"]) < 0.0
        and float(q_theory_diagnostic["rel_vs_alpha_exact_q_theory"]) > 0.9
    )
    all_extra_labels_phase3_carried_now = all(
        bool(q_surface_diagnostics[label]["legacy_phase3_sideband"]) for label in qext_labels
    )
    best_extra_label_legacy_phase3_now = bool(
        best_extra_label_vs_alpha_exact is not None
        and q_surface_diagnostics[best_extra_label_vs_alpha_exact]["legacy_phase3_sideband"]
    )
    canonical_extra_q_rescue_available_now = False
    phase3_sideband_carryover_only_now = bool(
        q_theory_failure_preserved_now
        and all_extra_labels_phase3_carried_now
        and best_extra_label_legacy_phase3_now
        and not canonical_extra_q_rescue_available_now
    )
    numeric_rerun_available_now = bool(
        qext_pack["selected_extension_solver_side_extra_q_range_helper_available_now"]
        and bool(alpha_qext_pack)
    )

    return {
        "selected_extension_label": qext_pack["selected_extension_label"],
        "solver_side_deformation_label": qext_pack["solver_side_deformation_label"],
        "source_materialization_label": qext_pack["source_materialization_label"],
        "retained_q_window": qext_pack["retained_q_window"],
        "q_ext_ind_window": qext_pack["q_ext_ind_window"],
        "q_aug_window": qext_pack["q_aug_window"],
        "blind_F_qext_pack": qext_pack["F_blind_qext_pack"],
        "alpha_blind_qext_pack": alpha_qext_pack,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "alpha_target": alpha_target,
        "scalar_q_exact_over_m0": scalar_q_exact_over_m0,
        "q_theory_over_m0": q_theory_over_m0,
        "q_surface_diagnostics": q_surface_diagnostics,
        "best_extra_label_vs_alpha_exact": best_extra_label_vs_alpha_exact,
        "best_extra_label_vs_alpha_target": best_extra_label_vs_alpha_target,
        "best_extra_label_vs_q_exact": best_extra_label_vs_q_exact,
        "best_extra_alpha_exact_residual": best_extra_alpha_exact_residual,
        "best_extra_alpha_target_residual": best_extra_alpha_target_residual,
        "best_extra_q_exact_gap": best_extra_q_exact_gap,
        "q_theory_failure_preserved_now": q_theory_failure_preserved_now,
        "all_extra_labels_phase3_carried_now": all_extra_labels_phase3_carried_now,
        "best_extra_label_legacy_phase3_now": best_extra_label_legacy_phase3_now,
        "canonical_extra_q_rescue_available_now": canonical_extra_q_rescue_available_now,
        "phase3_sideband_carryover_only_now": phase3_sideband_carryover_only_now,
        "selected_extension_solver_side_extra_q_range_numeric_rerun_available_now": numeric_rerun_available_now,
    }


# 関数: helper 実行時に short summary を返す。

def main() -> None:
    """Run the extra-q numeric rerun diagnostic and print a compact summary."""
    pack = build_selected_extension_solver_side_extra_q_range_numeric_rerun_pack()
    summary = {
        "selected_extension_label": pack["selected_extension_label"],
        "source_materialization_label": pack["source_materialization_label"],
        "q_exact_over_m0": pack["scalar_q_exact_over_m0"],
        "q_theory_over_m0": pack["q_theory_over_m0"],
        "best_extra_label_vs_alpha_exact": pack["best_extra_label_vs_alpha_exact"],
        "best_extra_alpha_exact_residual": pack["best_extra_alpha_exact_residual"],
        "best_extra_q_exact_gap": pack["best_extra_q_exact_gap"],
        "q_theory_failure_preserved_now": pack["q_theory_failure_preserved_now"],
        "phase3_sideband_carryover_only_now": pack["phase3_sideband_carryover_only_now"],
        "canonical_extra_q_rescue_available_now": pack["canonical_extra_q_rescue_available_now"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper summary を出力する。

if __name__ == "__main__":
    main()
