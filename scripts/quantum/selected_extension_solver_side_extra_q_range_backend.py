#!/usr/bin/env python3
"""
Materialize the selected-extension extra-q source-materialization output pack.

This helper is the implementation counterpart of `.5359-.5366`. It extends the
already materialized retained-q deformation pack on `Sigma_*^(pilot-HS)` by
lifting explicit extra-q checkpoints that are already frozen in the retained
blind-vector numeric-evaluation surface.

The current helper does not claim a new solver-derived positive rescue. It
builds the first actual helper-backed surface

    O_qext_sel^(pilot-HS)[Q_aug]
      = {K_eff^(pilot-HS,qext)[Q_aug],
         Z_eff^(pilot-HS,qext,T)[Q_aug],
         F_blind^(pilot-HS,qext)[Q_aug],
         alpha_blind^(pilot-HS,q_theory),
         Delta_qext_sel^(pilot-HS)}

with

    Q_aug^(pilot-HS) = Q_ret union Q_ext^(ind)

while preserving all retained-q checkpoints exactly.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.selected_extension_solver_side_deformation_backend import (
    build_deformation_effective_kernel_pack,
)
from scripts.quantum.selected_extension_solver_side_deformation_backend import (
    build_selected_extension_solver_side_deformation_pack,
)
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_"
    "blind_vector_observable_gate_numeric_evaluation_metrics.json"
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


# 関数: blind sideband metrics から explicit な extra-q checkpoints を抽出する。

def build_independent_extra_q_support(
    summary: dict,
    retained_q_window: dict[str, float],
) -> dict[str, dict[str, float | str]]:
    """Return the explicit extra-q checkpoints carried by the blind numeric-evaluation surface."""
    candidates = [
        (
            "best_global_signed_q",
            float(summary["best_global_signed_q"]),
            float(summary["best_global_signed_F"]),
            "best_global_signed_surface",
        ),
        (
            "local_band_abs_best_q",
            float(summary["local_band_abs_best_q"]),
            float(summary["local_band_abs_best_F"]),
            "local_abs_surface",
        ),
        (
            "local_band_signed_best_q",
            float(summary["local_band_signed_best_q"]),
            float(summary["local_band_signed_best_F"]),
            "local_signed_surface",
        ),
    ]
    retained_values = (
        float(retained_q_window["zero"]),
        float(retained_q_window["q_theory_over_m0"]),
        float(retained_q_window["m0"]),
    )
    support: dict[str, dict[str, float | str]] = {}
    seen_values: list[float] = []
    for label, q_value, f_value, source_tag in sorted(candidates, key=lambda item: item[1]):
        if any(math.isclose(q_value, retained_q, rel_tol=0.0, abs_tol=1.0e-12) for retained_q in retained_values):
            continue

        if any(math.isclose(q_value, seen_q, rel_tol=0.0, abs_tol=1.0e-12) for seen_q in seen_values):
            continue

        support[label] = {
            "q_over_m0": q_value,
            "F_blind_value": f_value,
            "transverse_scalar_value": f_value,
            "source_tag": source_tag,
            "source_phase": "phase3_blind_numeric_evaluation",
        }
        seen_values.append(q_value)

    return support


# 関数: retained surface と extra-q support を一つの scalar pack に束ねる。

def build_augmented_scalar_pack(
    retained_pack: dict[str, float],
    extra_support: dict[str, dict[str, float | str]],
) -> dict[str, float]:
    """Return the augmented transverse-scalar pack on Q_aug."""
    augmented = {label: float(value) for label, value in retained_pack.items()}
    for label, payload in extra_support.items():
        augmented[label] = float(payload["transverse_scalar_value"])

    return augmented


# 関数: actual helper-backed extra-q source-materialization pack を構成する。

def build_selected_extension_solver_side_extra_q_range_pack(
    ell_values: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Materialize one helper-backed extra-q source surface on the selected extension."""
    require(PHASE3_EVAL)

    deformation_pack = build_selected_extension_solver_side_deformation_pack(
        ell_values=ell_values
    )
    phase3_summary = read_json(PHASE3_EVAL)["summary"]
    extra_support = build_independent_extra_q_support(
        phase3_summary,
        deformation_pack["retained_q_window"],
    )

    z_eff_qext_transverse_scalar = build_augmented_scalar_pack(
        deformation_pack["Z_eff_deform_transverse_scalar_pack"],
        extra_support,
    )
    k_eff_qext = build_deformation_effective_kernel_pack(z_eff_qext_transverse_scalar)
    f_blind_qext = dict(z_eff_qext_transverse_scalar)

    retained_surface_preserved_now = all(
        math.isclose(
            float(f_blind_qext[label]),
            float(deformation_pack["F_blind_deform_pack"][label]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for label in ("zero", "q_theory_over_m0", "m0")
    )
    q_ext_ind_nonempty_now = bool(extra_support)
    q_aug_materialized_now = bool(
        {"zero", "q_theory_over_m0", "m0"} <= set(f_blind_qext)
        and q_ext_ind_nonempty_now
    )
    source_materialization_helper_available_now = bool(
        retained_surface_preserved_now and q_aug_materialized_now
    )

    q_aug_window = {
        **{
            "zero": float(deformation_pack["retained_q_window"]["zero"]),
            "q_theory_over_m0": float(
                deformation_pack["retained_q_window"]["q_theory_over_m0"]
            ),
            "m0": float(deformation_pack["retained_q_window"]["m0"]),
        },
        **{label: float(payload["q_over_m0"]) for label, payload in extra_support.items()},
    }
    q_ext_ind_window = {
        label: float(payload["q_over_m0"]) for label, payload in extra_support.items()
    }

    return {
        "selected_extension_label": deformation_pack["selected_extension_label"],
        "solver_side_deformation_label": deformation_pack[
            "solver_side_deformation_label"
        ],
        "source_materialization_label": "R_qsrc^(helper_impl,pilot-HS)",
        "retained_q_window": deformation_pack["retained_q_window"],
        "q_ext_ind_window": q_ext_ind_window,
        "q_aug_window": q_aug_window,
        "K_eff_qext_transverse_projector_pack": k_eff_qext,
        "Z_eff_qext_transverse_scalar_pack": z_eff_qext_transverse_scalar,
        "F_blind_qext_pack": f_blind_qext,
        "alpha_blind_qext_at_q_theory": float(
            deformation_pack["alpha_blind_deform_at_q_theory"]
        ),
        "alpha_exact_at_q_theory": float(deformation_pack["alpha_exact_at_q_theory"]),
        "delta_alpha_sel_qext_exact": float(
            deformation_pack["delta_alpha_sel_deform_exact"]
        ),
        "relative_exact_residual_qext": float(
            deformation_pack["relative_exact_residual_deform"]
        ),
        "retained_surface_preserved_now": retained_surface_preserved_now,
        "q_ext_ind_nonempty_now": q_ext_ind_nonempty_now,
        "q_aug_materialized_now": q_aug_materialized_now,
        "selected_extension_solver_side_extra_q_range_helper_available_now": source_materialization_helper_available_now,
        "phase3_sideband_source_summary": {
            "best_global_signed_q": float(phase3_summary["best_global_signed_q"]),
            "best_global_signed_F": float(phase3_summary["best_global_signed_F"]),
            "local_band_signed_best_q": float(
                phase3_summary["local_band_signed_best_q"]
            ),
            "local_band_signed_best_F": float(
                phase3_summary["local_band_signed_best_F"]
            ),
            "local_band_abs_best_q": float(phase3_summary["local_band_abs_best_q"]),
            "local_band_abs_best_F": float(phase3_summary["local_band_abs_best_F"]),
        },
        "evidence_samples": deformation_pack["evidence_samples"],
        "best_exact_match_or_none": deformation_pack["best_exact_match_or_none"],
        "retained_anchor_row": deformation_pack["retained_anchor_row"],
    }


# 関数: helper 実行時に short summary を返す。

def main() -> None:
    """Run the selected-extension extra-q helper and print a compact summary."""
    pack = build_selected_extension_solver_side_extra_q_range_pack()
    summary = {
        "selected_extension_label": pack["selected_extension_label"],
        "source_materialization_label": pack["source_materialization_label"],
        "q_ext_ind_window": pack["q_ext_ind_window"],
        "q_aug_window": pack["q_aug_window"],
        "F_blind_qext_pack": pack["F_blind_qext_pack"],
        "alpha_blind_qext_at_q_theory": pack["alpha_blind_qext_at_q_theory"],
        "delta_alpha_sel_qext_exact": pack["delta_alpha_sel_qext_exact"],
        "relative_exact_residual_qext": pack["relative_exact_residual_qext"],
        "retained_surface_preserved_now": pack["retained_surface_preserved_now"],
        "q_ext_ind_nonempty_now": pack["q_ext_ind_nonempty_now"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper summary を出力する。

if __name__ == "__main__":
    main()
