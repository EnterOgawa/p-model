#!/usr/bin/env python3
"""Generate 8.7.56.2267-.2270 hybrid seventh extreme ultra-farther registry artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_2259 as prior
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


STEM = build_compact_artifact_stem(
    "8.7.56.2267-2270",
    "harmonic_hybrid_s7_s8_extreme_ultra_registry",
    prefix="q",
)


# 関数: 前段 registry module の route 定数を current branch 向けに上書きする。
def configure() -> None:
    """Retarget the prior hybrid registry module to `.2267-.2270`."""
    prior.PRIOR_GATE = build_metrics_paths(
        prior.PUBLIC_OUT,
        build_compact_artifact_stem(
            "8.7.56.2263-2266",
            "harmonic_hybrid_s7_s8_extreme_ultra_fast",
            prefix="q",
        ),
        "declaration_gate",
    )["json"]
    prior.STEP_TAG = "8.7.56.2267-2270"
    prior.STEP_NAME = (
        "Trial-2 numeric alpha vector Q-ball form-factor hybrid seventh "
        "extreme ultra-farther registry refresh"
    )
    prior.STEM = STEM
    prior.PRIOR_CLASS = (
        "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s7_retained_1572864_next"
    )
    prior.BRANCH_CLASS_SEVENTH = (
        "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s7_retained_1671168_next"
    )
    prior.BRANCH_CLASS_EIGHTH = (
        "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s8_promoted_1671168_next"
    )
    prior.BRANCH_CLASS_RESET = (
        "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s7s8_extreme_ultra_exhausted_pack_update_next"
    )
    prior.NEXT_ROUTE_NAME_SEVENTH = (
        "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s7_super_extreme_ultra_farther_audit"
    )
    prior.NEXT_ROUTE_NAME_EIGHTH = (
        "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s8_super_extreme_ultra_farther_audit"
    )
    prior.NEXT_ROUTE_NAME_RESET = (
        "trial2_numeric_alpha_vector_qball_form_factor_hybrid_extreme_ultra_pack_update_review"
    )
    prior.NEXT_ROUTE = "8.7.56.2271"
    prior.FOLLOWUP_ROUTE_NAME_SEVENTH = (
        "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s7_super_extreme_registry_refresh"
    )
    prior.FOLLOWUP_ROUTE_NAME_EIGHTH = (
        "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s8_super_extreme_registry_refresh"
    )
    prior.FOLLOWUP_ROUTE_NAME_RESET = (
        "trial2_numeric_alpha_vector_qball_form_factor_hybrid_extreme_ultra_pack_update_registry"
    )
    prior.FOLLOWUP_ROUTE = "8.7.56.2275"


# 関数: `.2267-.2270` を実行する。

def main() -> None:
    """Run the retargeted hybrid seventh extreme ultra-farther registry refresh."""
    configure()
    prior.main()


if __name__ == "__main__":
    main()
