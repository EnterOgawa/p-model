#!/usr/bin/env python3
"""Audit why the current two-surface table splits between CODATA and P-model.

Purpose:
    The first actual multi-observable comparison now has one CODATA win
    (Hydrogen 1S-2S gross structure) and one retained P-model win
    (H I 21 cm hyperfine Fermi baseline). This backend localizes the split to
    the surface-implied effective-alpha values read by those two baselines.

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_first_multi_observable_comparison_refresh_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5947-.5950` wrappers
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_first_multi_observable_comparison_refresh_backend import (
    build_trial2_first_multi_observable_comparison_refresh_pack,
)
from scripts.quantum.trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend import (
    build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack,
    hydrogen_hyperfine_fermi_frequency_hz,
)
from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
    hydrogen_1s2s_gross_frequency_hz,
)


ALPHA_LABEL_ORDER = [
    "alpha_P_frozen",
    "alpha_common",
    "alpha_P_4D_can",
    "alpha_P_4D_vertex",
    "alpha_CODATA",
]


# 関数: monotone alpha^2 surface から implied alpha を逆算する。
def invert_alpha_squared_surface(observed_hz: float) -> float:
    """Return the implied alpha on a quadratic absolute surface."""
    return float(math.sqrt(observed_hz / hydrogen_1s2s_gross_frequency_hz(1.0)))


# 関数: monotone alpha^4 surface から implied alpha を逆算する。

def invert_alpha_quartic_surface(observed_hz: float) -> float:
    """Return the implied alpha on a quartic absolute surface."""
    return float((observed_hz / hydrogen_hyperfine_fermi_frequency_hz(1.0)) ** 0.25)


# 関数: 1本の checkpoint row を surface-implied alpha に対して評価する。

def build_gap_row(*, alpha_label: str, alpha_value: float, alpha_eff_1s2s: float, alpha_eff_hfs: float) -> dict:
    """Return one gap row against both surface-implied effective alphas."""
    gap_1s2s = float((alpha_value - alpha_eff_1s2s) / alpha_eff_1s2s)
    gap_hfs = float((alpha_value - alpha_eff_hfs) / alpha_eff_hfs)
    return {
        "alpha_label": alpha_label,
        "alpha_value": float(alpha_value),
        "relative_gap_vs_alpha_eff_1s2s": gap_1s2s,
        "relative_gap_vs_alpha_eff_hfs": gap_hfs,
        "abs_relative_gap_vs_alpha_eff_1s2s": float(abs(gap_1s2s)),
        "abs_relative_gap_vs_alpha_eff_hfs": float(abs(gap_hfs)),
    }


# 関数: `.5947-.5950` 用の attribution pack を返す。

def build_trial2_hyperfine_attribution_split_audit_pack() -> dict:
    """Return the retained hyperfine attribution-split audit pack."""
    multi_pack = build_trial2_first_multi_observable_comparison_refresh_pack()
    first_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    hyperfine_pack = build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack()

    alpha_constants = dict(first_pack["alpha_constants"])
    observed_1s2s = float(first_pack["summary"]["hydrogen_1s2s_observed_hz"])
    observed_hfs = float(hyperfine_pack["summary"]["observed_hz"])

    alpha_eff_1s2s = invert_alpha_squared_surface(observed_1s2s)
    alpha_eff_hfs = invert_alpha_quartic_surface(observed_hfs)
    relative_split = float((alpha_eff_hfs - alpha_eff_1s2s) / alpha_eff_1s2s)

    gap_rows = [
        build_gap_row(
            alpha_label=label,
            alpha_value=float(alpha_constants[label]),
            alpha_eff_1s2s=alpha_eff_1s2s,
            alpha_eff_hfs=alpha_eff_hfs,
        )
        for label in ALPHA_LABEL_ORDER
    ]

    closest_to_1s2s = min(gap_rows, key=lambda row: float(row["abs_relative_gap_vs_alpha_eff_1s2s"]))
    closest_to_hfs = min(gap_rows, key=lambda row: float(row["abs_relative_gap_vs_alpha_eff_hfs"]))

    return {
        "surface_effective_alphas": {
            "hydrogen_1s2s_gross_structure_baseline": alpha_eff_1s2s,
            "hydrogen_hyperfine_21cm_fermi_baseline": alpha_eff_hfs,
        },
        "checkpoint_gap_rows": gap_rows,
        "summary": {
            "split_watch_verdict_now": bool(multi_pack["summary"]["split_watch_verdict_now"]),
            "alpha_eff_1s2s": alpha_eff_1s2s,
            "alpha_eff_hfs": alpha_eff_hfs,
            "effective_alpha_split_relative": relative_split,
            "closest_to_1s2s_alpha_label": str(closest_to_1s2s["alpha_label"]),
            "closest_to_1s2s_relative_gap": float(closest_to_1s2s["relative_gap_vs_alpha_eff_1s2s"]),
            "closest_to_hfs_alpha_label": str(closest_to_hfs["alpha_label"]),
            "closest_to_hfs_relative_gap": float(closest_to_hfs["relative_gap_vs_alpha_eff_hfs"]),
            "hyperfine_attribution_split_localized_now": True,
            "attribution_reading": (
                "the split is localized to different surface-implied effective-alpha "
                "values rather than to one unresolved scorekeeping artifact"
            ),
        },
        "trial2_hyperfine_attribution_split_localized_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the hyperfine attribution backend directly."""
    pack = build_trial2_hyperfine_attribution_split_audit_pack()
    summary = pack["summary"]
    print("[trial2_hyperfine_attribution_split_audit_backend]")
    print(f"  alpha_eff_1s2s = {summary['alpha_eff_1s2s']}")
    print(f"  alpha_eff_hfs = {summary['alpha_eff_hfs']}")
    print(f"  effective_alpha_split_relative = {summary['effective_alpha_split_relative']}")
    print(f"  closest_to_1s2s_alpha_label = {summary['closest_to_1s2s_alpha_label']}")
    print(f"  closest_to_hfs_alpha_label = {summary['closest_to_hfs_alpha_label']}")


if __name__ == "__main__":
    main()
