#!/usr/bin/env python3
"""Audit whether Halpha fine structure can be promoted to a native surface now.

Purpose:
    Trial-2 primary observable comparison is constrained by the absolute rule
    "P-model formula x P-model alpha". Under that rule, Hydrogen Halpha
    fine-structure can be promoted only if the current public canon already
    exposes one honest relativistic bound-state bridge from the Part III-A
    envelope sector plus the adopted-U(1) Coulomb sector to the retained
    reduced-mass Dirac-Coulomb baseline.

    This backend cuts that question mechanically. The retained fine-structure
    baseline itself already exists, but the public-canonical relativistic bridge
    may still be absent.

Inputs:
    - scripts/quantum/trial2_hydrogen_fine_structure_absolute_alpha_formula_materialization_backend.py

Outputs:
    - One in-memory audit pack consumed by `.6015-.6018` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_hydrogen_fine_structure_absolute_alpha_formula_materialization_backend import (
    build_trial2_hydrogen_fine_structure_absolute_alpha_formula_pack,
)


# 関数: `.6015-.6018` 用の Halpha native-materialization pack を返す。
def build_trial2_native_relativistic_halpha_surface_materialization_pack() -> dict:
    """Return the retained Halpha native-materialization audit pack."""
    fine_structure_pack = build_trial2_hydrogen_fine_structure_absolute_alpha_formula_pack()
    surface = dict(fine_structure_pack["surface"])

    part3a_positive_frequency_kg_public_now = True
    part3a_schrodinger_envelope_public_now = True
    part3a_relativistic_bound_state_bridge_public_now = False
    adopted_u1_coulomb_route_public_now = True
    adopted_u1_relativistic_bound_state_bridge_public_now = False

    native_relativistic_surface_ready_now = bool(
        surface["current_alpha_rerun_ready_now"]
        and part3a_relativistic_bound_state_bridge_public_now
        and adopted_u1_relativistic_bound_state_bridge_public_now
    )

    return {
        "surface": surface,
        "bridge_checkpoints": {
            "part3a_positive_frequency_kg_public_now": part3a_positive_frequency_kg_public_now,
            "part3a_schrodinger_envelope_public_now": part3a_schrodinger_envelope_public_now,
            "part3a_relativistic_bound_state_bridge_public_now": (
                part3a_relativistic_bound_state_bridge_public_now
            ),
            "adopted_u1_coulomb_route_public_now": adopted_u1_coulomb_route_public_now,
            "adopted_u1_relativistic_bound_state_bridge_public_now": (
                adopted_u1_relativistic_bound_state_bridge_public_now
            ),
        },
        "summary": {
            "selected_surface_id": str(surface["surface_id"]),
            "selected_surface_label": str(surface["label"]),
            "diagnostic_surface_retained_now": True,
            "native_relativistic_surface_ready_now": bool(native_relativistic_surface_ready_now),
            "best_overall_alpha_label": str(
                fine_structure_pack["summary"]["selected_best_overall_alpha_label"]
            ),
            "best_overall_relative_error_vs_observed": float(
                fine_structure_pack["summary"]["selected_best_overall_relative_error_vs_observed"]
            ),
            "best_pmodel_alpha_label": str(
                fine_structure_pack["summary"]["selected_best_pmodel_alpha_label"]
            ),
            "best_pmodel_relative_error_vs_observed": float(
                fine_structure_pack["summary"]["selected_best_pmodel_relative_error_vs_observed"]
            ),
            "current_honest_reading": (
                "The retained Halpha fine-structure formula exists only as a "
                "reduced-mass Dirac-Coulomb diagnostic. Part III-A currently "
                "publishes the positive-frequency KG -> Schr envelope and the "
                "adopted-U(1) Coulomb route, but not one public-canonical "
                "relative-relativistic bound-state bridge."
            ),
        },
        "trial2_native_relativistic_halpha_surface_materialization_completed_now": True,
        "trial2_native_relativistic_halpha_surface_ready_now": bool(
            native_relativistic_surface_ready_now
        ),
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the native relativistic Halpha audit backend directly."""
    pack = build_trial2_native_relativistic_halpha_surface_materialization_pack()
    summary = pack["summary"]
    print("[trial2_native_relativistic_halpha_surface_materialization_backend]")
    print(f"  selected_surface_id = {summary['selected_surface_id']}")
    print(
        "  native_relativistic_surface_ready_now = "
        f"{summary['native_relativistic_surface_ready_now']}"
    )
    print(
        "  best_overall_alpha_label = "
        f"{summary['best_overall_alpha_label']}"
    )


if __name__ == "__main__":
    main()
