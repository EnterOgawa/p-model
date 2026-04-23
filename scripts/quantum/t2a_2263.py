#!/usr/bin/env python3
"""Generate 8.7.56.2263-.2266 hybrid seventh/eighth extreme ultra-farther artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_2255 as prior
from scripts.utils.windows_length_policy import build_compact_artifact_stem


STEM = build_compact_artifact_stem(
    "8.7.56.2263-2266",
    "harmonic_hybrid_s7_s8_extreme_ultra_fast",
    prefix="q",
)
REGISTRY_ALIAS = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_2263_2266_harmonic_hybrid_s7_registry_alias.json"
)


# 関数: 前段 audit module の route 定数を current branch 向けに上書きする。
def build_registry_alias() -> None:
    """Create a shim registry file with the key name expected by `t2a_2255`."""
    source_path = (
        prior.base.PUBLIC_OUT
        / "q_8_7_56_2259_2262_harmonic_hybrid_s7_s8_ultra_registry_declaration_gate_metrics.json"
    )
    payload = prior.base.sign_base.read_json(source_path)
    payload["summary"][
        "gate_a_same_seventh_piecewise_validation_to_1474560_retained"
    ] = payload["summary"]["gate_a_same_seventh_piecewise_validation_to_1572864_retained"]
    REGISTRY_ALIAS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# 関数: 前段 audit module の route 定数を current branch 向けに上書きする。

def configure() -> None:
    """Retarget the prior hybrid ultra-farther audit module to `.2263-.2266`."""
    build_registry_alias()
    prior.PRIOR_AUDIT = (
        prior.base.PUBLIC_OUT
        / "q_8_7_56_2255_2258_harmonic_hybrid_s7_s8_ultra_fast_declaration_gate_metrics.json"
    )
    prior.PRIOR_REGISTRY = REGISTRY_ALIAS
    prior.STEP_TAG = "8.7.56.2263-2266"
    prior.STEP_NAME = (
        "Trial-2 numeric alpha vector Q-ball form-factor hybrid seventh/eighth "
        "extreme ultra-farther audit"
    )
    prior.STEM = STEM
    prior.base.STEM = STEM
    prior.PRIOR_CLASS = (
        "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s7_retained_1572864_next"
    )
    prior.BRANCH_CLASS = (
        "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s7_s8_extreme_ultra_fast_gate"
    )
    prior.NEXT_ROUTE_NAME = (
        "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s7s8_extreme_ultra_registry"
    )
    prior.NEXT_ROUTE = "8.7.56.2267"
    prior.FOLLOWUP_ROUTE_NAME = (
        "trial2_numeric_alpha_vector_qball_form_factor_hybrid_selected_super_extreme_ultra_farther_audit"
    )
    prior.FOLLOWUP_ROUTE = "8.7.56.2271"
    prior.FARTHER_BANDS = [
        (1572865, 1581056),
        (1581057, 1589248),
        (1589249, 1597440),
        (1597441, 1605632),
        (1605633, 1613824),
        (1613825, 1622016),
        (1622017, 1630208),
        (1630209, 1638400),
        (1638401, 1646592),
        (1646593, 1654784),
        (1654785, 1662976),
        (1662977, 1671168),
    ]
    prior.FIRST_HOLDOUT = prior.FARTHER_BANDS[:4]
    prior.FIRST_MONITOR = prior.FARTHER_BANDS[4:]
    prior.RESERVE_FIT = prior.FARTHER_BANDS[:4]
    prior.RESERVE_HOLDOUT = prior.FARTHER_BANDS[4:8]
    prior.RESERVE_MONITOR = prior.FARTHER_BANDS[8:]


# 関数: `.2263-.2266` を実行する。

def main() -> None:
    """Run the retargeted hybrid seventh/eighth extreme ultra-farther audit."""
    configure()
    prior.main()


if __name__ == "__main__":
    main()
