#!/usr/bin/env python3
"""
Materialize the selected-extension solver-recompute output pack.

This helper is the implementation counterpart of `.5231-.5238`. It reuses the
already materialized selected-extension backend pack on `Sigma_*^(pilot-HS)` and
lifts it into the explicit retained-q output objects

    O_recomp_sel^(pilot-HS)
      = {K_eff^(pilot-HS,recomp)[Q_ret],
         Z_eff^(pilot-HS,recomp,T)[Q_ret],
         F_blind^(pilot-HS,recomp)[Q_ret],
         alpha_blind^(pilot-HS,recomp)(q_theory),
         delta_alpha_sel^(pilot-HS,recomp)}.

The current implementation keeps the retained-q checkpoint semantics explicit
and exposes the first materialized recompute pack that later branches can audit
for genuine solver-side deformation versus simple retained replay.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.quantum.blind_vector_selected_extension_backend import (
    build_selected_extension_backend_pack,
)


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
SCALAR_TARGET = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_"
    "reconciliation_review_numeric_evaluation_metrics.json"
)


# 関数: 必須 input artifact の存在を確認する。
def require(path: Path) -> None:
    """Abort immediately when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON artifact into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: retained blind checkpoint から retained-q scalar pack を作る。

def build_transverse_scalar_pack(blind_target_keys: dict) -> dict[str, float]:
    """Return the retained-q transverse-scalar values used by the recompute pack."""
    return {
        "zero": float(blind_target_keys["blind_F_at_zero"]),
        "q_theory_over_m0": float(blind_target_keys["blind_F_at_q_theory"]),
        "m0": float(blind_target_keys["blind_F_at_m0"]),
    }


# 関数: retained-q scalar pack を projector-coefficient 付き kernel pack へ写す。

def build_effective_kernel_pack(z_eff_pack: dict[str, float]) -> dict[str, dict[str, float | str]]:
    """Return one projector-sector effective-kernel pack on the retained q window."""
    return {
        q_label: {
            "transverse_projector_sector": "Pi_T",
            "scalar_coefficient": float(value),
        }
        for q_label, value in z_eff_pack.items()
    }


# 関数: selected-extension solver-recompute output pack を materialize する。

def build_selected_extension_solver_recompute_pack(
    ell_values: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Materialize one retained-q solver-recompute pack on Sigma_*^(pilot-HS)."""
    require(SCALAR_TARGET)

    backend_pack = build_selected_extension_backend_pack(ell_values=ell_values)
    scalar_summary = read_json(SCALAR_TARGET)["summary"]

    z_eff_recomp_transverse_scalar = build_transverse_scalar_pack(
        backend_pack["blind_target_keys"]
    )
    k_eff_recomp = build_effective_kernel_pack(z_eff_recomp_transverse_scalar)
    f_blind_recomp = dict(z_eff_recomp_transverse_scalar)

    alpha_exact_at_q_theory = float(scalar_summary["alpha_exact_at_q_theory"])
    alpha_blind_recomp_at_q_theory = float(
        backend_pack["blind_target_keys"]["blind_alpha_at_q_theory"]
    )
    delta_alpha_sel_recomp_exact = float(
        alpha_blind_recomp_at_q_theory - alpha_exact_at_q_theory
    )
    relative_exact_residual_recomp = float(
        abs(delta_alpha_sel_recomp_exact) / alpha_exact_at_q_theory
    )

    preserves_retained_phase3_checkpoint_now = bool(
        math.isclose(
            float(f_blind_recomp["zero"]),
            float(backend_pack["blind_target_keys"]["blind_F_at_zero"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(f_blind_recomp["q_theory_over_m0"]),
            float(backend_pack["blind_target_keys"]["blind_F_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(f_blind_recomp["m0"]),
            float(backend_pack["blind_target_keys"]["blind_F_at_m0"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            alpha_blind_recomp_at_q_theory,
            float(backend_pack["blind_target_keys"]["blind_alpha_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )

    return {
        "selected_extension_label": backend_pack["selected_extension_label"],
        "retained_q_window": backend_pack["retained_q_window"],
        "K_eff_recomp_transverse_projector_pack": k_eff_recomp,
        "Z_eff_recomp_transverse_scalar_pack": z_eff_recomp_transverse_scalar,
        "F_blind_recomp_pack": f_blind_recomp,
        "alpha_blind_recomp_at_q_theory": alpha_blind_recomp_at_q_theory,
        "alpha_exact_at_q_theory": alpha_exact_at_q_theory,
        "delta_alpha_sel_recomp_exact": delta_alpha_sel_recomp_exact,
        "relative_exact_residual_recomp": relative_exact_residual_recomp,
        "preserves_retained_phase3_checkpoint_now": preserves_retained_phase3_checkpoint_now,
        "ell_scan_counts": backend_pack["ell_scan_counts"],
        "base_mode_counts": backend_pack["base_mode_counts"],
        "exact_ladder_row_count": int(backend_pack["exact_ladder_row_count"]),
        "comparison_row_count": int(backend_pack["comparison_row_count"]),
        "best_exact_match_or_none": backend_pack["best_exact_match"],
        "retained_anchor_row": backend_pack["retained_anchor_row"],
        "evidence_samples": backend_pack["evidence_samples"],
    }


# 関数: helper 実行時に short summary を返す。

def main() -> None:
    """Run the selected-extension solver-recompute helper and print a compact summary."""
    pack = build_selected_extension_solver_recompute_pack()
    summary = {
        "selected_extension_label": pack["selected_extension_label"],
        "retained_q_window": pack["retained_q_window"],
        "F_blind_recomp_pack": pack["F_blind_recomp_pack"],
        "alpha_blind_recomp_at_q_theory": pack["alpha_blind_recomp_at_q_theory"],
        "delta_alpha_sel_recomp_exact": pack["delta_alpha_sel_recomp_exact"],
        "relative_exact_residual_recomp": pack["relative_exact_residual_recomp"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper summary を出力する。

if __name__ == "__main__":
    main()
