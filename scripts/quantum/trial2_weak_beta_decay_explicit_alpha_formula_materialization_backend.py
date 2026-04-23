#!/usr/bin/env python3
"""Audit whether the weak beta-decay pack exposes a fine-structure-alpha formula.

Purpose:
    Trial-2 observable comparison needs explicit `alpha -> observable` formulas.
    The weak beta-decay Route A/B pack is quantitative and independent, but the
    current public implementation still may not expose fine-structure alpha as a
    deterministic rerun lever. This backend fixes that verdict.

Inputs:
    - output/public/quantum/weak_interaction_beta_decay_route_ab_audit.json
    - scripts/quantum/weak_interaction_beta_decay_route_ab_audit.py
    - scripts/quantum/weak_interaction_beta_decay_route_b_standalone_audit.py

Outputs:
    - One in-memory audit pack consumed by `.5915-.5918` wrappers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_weak_sector_alpha_dependency_materialization_backend import (
    build_trial2_weak_sector_alpha_materialization_pack,
)


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
ROUTE_AB_SCRIPT = ROOT / "scripts" / "quantum" / "weak_interaction_beta_decay_route_ab_audit.py"
ROUTE_B_STANDALONE = ROOT / "scripts" / "quantum" / "weak_interaction_beta_decay_route_b_standalone_audit.py"

ALPHA_P_FROZEN = 0.007302943961943229
ALPHA_P_4D_CAN = 0.0072988143426522215
ALPHA_CODATA = 0.0072973525643


# 関数: 1 本の UTF-8 JSON を読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: 1 本の UTF-8 text を読む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: internal alpha collision evidence を返す。

def build_internal_alpha_symbol_evidence() -> dict:
    """Return whether Route-B standalone uses `alpha` as an internal blend weight."""
    text = read_text(ROUTE_B_STANDALONE)
    patterns = [
        "return (1.0 - alpha) * q_sat + alpha * q_sign",
        "blend_eff = float(alpha)",
    ]
    hits = [pattern for pattern in patterns if pattern in text]
    return {
        "route_b_internal_alpha_symbol_collision_now": bool(hits),
        "matched_patterns": hits,
    }


# 関数: `.5915-.5918` 用の audit pack を返す。

def build_trial2_weak_beta_decay_explicit_alpha_formula_pack() -> dict:
    """Return the retained weak beta-decay explicit-alpha audit pack."""
    prior_pack = build_trial2_weak_sector_alpha_materialization_pack()
    route_ab_payload = read_json(PUBLIC_OUT / "weak_interaction_beta_decay_route_ab_audit.json")
    transition = str(route_ab_payload["decision"]["transition"])
    symbol_evidence = build_internal_alpha_symbol_evidence()

    surfaces = [
        {
            "surface_id": "weak_beta_decay_route_ab",
            "label": "Weak beta-decay Route A/B",
            "alpha_dependency_kind": "no_public_fine_structure_alpha_input",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": True,
            "primary_score_admissible_now": False,
            "selected_secondary_target_now": True,
            "notes": (
                "Route A/B is quantitative and independent, but the current public "
                "I/F exposes Q-value surrogates and closure gates only; fine-structure "
                "alpha does not enter as a deterministic rerun input."
            ),
            "route_ab_transition": transition,
        },
        {
            "surface_id": "weak_route_b_standalone_internal_alpha",
            "label": "Weak Route-B standalone internal alpha symbol",
            "alpha_dependency_kind": "internal_blend_symbol_not_fine_structure_alpha",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": False,
            "primary_score_admissible_now": False,
            "selected_secondary_target_now": False,
            "notes": (
                "The standalone Route-B script uses the symbol `alpha` as an internal "
                "blend weight for surrogate Q-value mixing, not as the fine-structure "
                "constant."
            ),
            "route_b_internal_alpha_symbol_collision_now": bool(
                symbol_evidence["route_b_internal_alpha_symbol_collision_now"]
            ),
            "matched_patterns": list(symbol_evidence["matched_patterns"]),
        },
    ]

    explicit_formula_ready_count = sum(
        1 for row in surfaces if row["primary_score_admissible_now"]
    )

    return {
        "alpha_constants": {
            "alpha_P_frozen": ALPHA_P_FROZEN,
            "alpha_P_4D_can": ALPHA_P_4D_CAN,
            "alpha_CODATA": ALPHA_CODATA,
        },
        "prior_weak_pack_summary": prior_pack["summary"],
        "surfaces": surfaces,
        "summary": {
            "weak_explicit_formula_surface_count": len(surfaces),
            "weak_explicit_formula_ready_count": explicit_formula_ready_count,
            "selected_secondary_target_ids": [
                str(row["surface_id"]) for row in surfaces if row["selected_secondary_target_now"]
            ],
            "route_ab_transition": transition,
            "route_b_internal_alpha_symbol_collision_now": bool(
                symbol_evidence["route_b_internal_alpha_symbol_collision_now"]
            ),
        },
        "trial2_weak_beta_decay_explicit_formula_materialized_now": False,
        "trial2_weak_beta_decay_primary_ready_now": False,
        "trial2_weak_beta_decay_negative_closeout_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the weak beta-decay explicit-alpha audit directly."""
    pack = build_trial2_weak_beta_decay_explicit_alpha_formula_pack()
    summary = pack["summary"]
    print("[trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend]")
    print(
        "  weak_explicit_formula_ready_count = "
        f"{summary['weak_explicit_formula_ready_count']}"
    )
    print(
        "  route_b_internal_alpha_symbol_collision_now = "
        f"{summary['route_b_internal_alpha_symbol_collision_now']}"
    )


if __name__ == "__main__":
    main()
