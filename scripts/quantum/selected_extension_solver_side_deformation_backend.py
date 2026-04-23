#!/usr/bin/env python3
"""
Materialize the selected-extension solver-side deformation output pack.

This helper is the implementation counterpart of `.5271-.5278`. It reuses the
already materialized selected-extension solver-recompute pack on
`Sigma_*^(pilot-HS)` and lifts it into the explicit deformation-labeled output
objects

    O_deform_sel^(pilot-HS)
      = {K_eff^(pilot-HS,deform)[Q_ret],
         Z_eff^(pilot-HS,deform,T)[Q_ret],
         F_blind^(pilot-HS,deform)[Q_ret],
         alpha_blind^(pilot-HS,deform)(q_theory),
         delta_alpha_sel^(pilot-HS,deform)}.

The current implementation keeps the retained-q checkpoint semantics explicit
and exposes the first materialized deformation pack that later branches can
audit for genuine solver-side deformation versus preserved replay.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.quantum.selected_extension_solver_recompute_backend import (
    build_selected_extension_solver_recompute_pack,
)


ROOT = Path(__file__).resolve().parents[2]


# 関数: retained-q scalar pack を projector-coefficient 付き deformation kernel pack へ写す。
def build_deformation_effective_kernel_pack(
    z_eff_pack: dict[str, float],
) -> dict[str, dict[str, float | str]]:
    """Return one projector-sector deformation-kernel pack on the retained q window."""
    return {
        q_label: {
            "transverse_projector_sector": "Pi_T",
            "scalar_coefficient": float(value),
            "deformation_contract_label": "D_solver_sel^(pilot-HS,recompute-retained)",
        }
        for q_label, value in z_eff_pack.items()
    }


# 関数: selected-extension solver-side deformation output pack を materialize する。

def build_selected_extension_solver_side_deformation_pack(
    ell_values: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Materialize one retained-q solver-side deformation pack on Sigma_*^(pilot-HS)."""
    recompute_pack = build_selected_extension_solver_recompute_pack(ell_values=ell_values)

    z_eff_deform_transverse_scalar = dict(
        recompute_pack["Z_eff_recomp_transverse_scalar_pack"]
    )
    k_eff_deform = build_deformation_effective_kernel_pack(
        z_eff_deform_transverse_scalar
    )
    f_blind_deform = dict(recompute_pack["F_blind_recomp_pack"])
    alpha_blind_deform_at_q_theory = float(
        recompute_pack["alpha_blind_recomp_at_q_theory"]
    )
    delta_alpha_sel_deform_exact = float(
        recompute_pack["delta_alpha_sel_recomp_exact"]
    )
    relative_exact_residual_deform = float(
        recompute_pack["relative_exact_residual_recomp"]
    )

    preserves_recompute_surface_now = bool(
        math.isclose(
            float(f_blind_deform["zero"]),
            float(recompute_pack["F_blind_recomp_pack"]["zero"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(f_blind_deform["q_theory_over_m0"]),
            float(recompute_pack["F_blind_recomp_pack"]["q_theory_over_m0"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(f_blind_deform["m0"]),
            float(recompute_pack["F_blind_recomp_pack"]["m0"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            alpha_blind_deform_at_q_theory,
            float(recompute_pack["alpha_blind_recomp_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            delta_alpha_sel_deform_exact,
            float(recompute_pack["delta_alpha_sel_recomp_exact"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            relative_exact_residual_deform,
            float(recompute_pack["relative_exact_residual_recomp"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )

    return {
        "selected_extension_label": recompute_pack["selected_extension_label"],
        "solver_side_deformation_label": "D_solver_sel^(pilot-HS,recompute-retained)",
        "deformation_contract_components": {
            "effective_kernel": "D_solver_sel^(K)",
            "internal_resolvent": "D_solver_sel^(G)",
            "retained_q_window": "D_solver_sel^(Qret)",
        },
        "retained_q_window": recompute_pack["retained_q_window"],
        "K_eff_deform_transverse_projector_pack": k_eff_deform,
        "Z_eff_deform_transverse_scalar_pack": z_eff_deform_transverse_scalar,
        "F_blind_deform_pack": f_blind_deform,
        "alpha_blind_deform_at_q_theory": alpha_blind_deform_at_q_theory,
        "alpha_exact_at_q_theory": float(recompute_pack["alpha_exact_at_q_theory"]),
        "delta_alpha_sel_deform_exact": delta_alpha_sel_deform_exact,
        "relative_exact_residual_deform": relative_exact_residual_deform,
        "preserves_recompute_surface_now": preserves_recompute_surface_now,
        "ell_scan_counts": recompute_pack["ell_scan_counts"],
        "base_mode_counts": recompute_pack["base_mode_counts"],
        "exact_ladder_row_count": int(recompute_pack["exact_ladder_row_count"]),
        "comparison_row_count": int(recompute_pack["comparison_row_count"]),
        "best_exact_match_or_none": recompute_pack["best_exact_match_or_none"],
        "retained_anchor_row": recompute_pack["retained_anchor_row"],
        "evidence_samples": recompute_pack["evidence_samples"],
    }


# 関数: helper 実行時に short summary を返す。

def main() -> None:
    """Run the selected-extension solver-side deformation helper and print a compact summary."""
    pack = build_selected_extension_solver_side_deformation_pack()
    summary = {
        "selected_extension_label": pack["selected_extension_label"],
        "solver_side_deformation_label": pack["solver_side_deformation_label"],
        "retained_q_window": pack["retained_q_window"],
        "F_blind_deform_pack": pack["F_blind_deform_pack"],
        "alpha_blind_deform_at_q_theory": pack["alpha_blind_deform_at_q_theory"],
        "delta_alpha_sel_deform_exact": pack["delta_alpha_sel_deform_exact"],
        "relative_exact_residual_deform": pack["relative_exact_residual_deform"],
        "preserves_recompute_surface_now": pack["preserves_recompute_surface_now"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper summary を出力する。

if __name__ == "__main__":
    main()
