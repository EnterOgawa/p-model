#!/usr/bin/env python3
"""Refresh the surface-attribution picture after the `g/2` hyperfine correction.

Purpose:
    The Fermi-only hyperfine surface produced a split between CODATA-like
    1S-2S and P-model-like 21 cm. Once the source-backed `g/2` correction is
    materialized, Trial-2 needs the corrected attribution picture:

      - what effective alpha does 1S-2S imply now?
      - what effective alpha does corrected hyperfine imply now?
      - does the old split shrink materially?

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_hyperfine_attribution_split_audit_backend.py
    - scripts/quantum/trial2_hyperfine_g2_correction_materialization_backend.py

Outputs:
    - One in-memory pack consumed by `.5963-.5966` wrappers
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_hyperfine_attribution_split_audit_backend import (
    ALPHA_LABEL_ORDER,
    build_trial2_hyperfine_attribution_split_audit_pack,
    build_gap_row,
)
from scripts.quantum.trial2_hyperfine_g2_correction_materialization_backend import (
    build_trial2_hyperfine_g2_correction_materialization_pack,
    hydrogen_hyperfine_g2_corrected_frequency_hz,
)
from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
    hydrogen_1s2s_gross_frequency_hz,
)


# 関数: alpha^2 surface から implied alpha を逆算する。
def invert_alpha_squared_surface(observed_hz: float) -> float:
    """Return the implied alpha on the gross-structure alpha^2 surface."""
    return float(math.sqrt(observed_hz / hydrogen_1s2s_gross_frequency_hz(1.0)))


# 関数: corrected alpha^4 surface から implied alpha を逆算する。

def invert_alpha_quartic_corrected_surface(observed_hz: float, *, g_over_2: float) -> float:
    """Return the implied alpha on the corrected hyperfine alpha^4 surface."""
    return float((observed_hz / hydrogen_hyperfine_g2_corrected_frequency_hz(1.0, g_over_2=g_over_2)) ** 0.25)


# 関数: `.5963-.5966` 用の corrected attribution pack を返す。

def build_trial2_hyperfine_corrected_attribution_refresh_pack() -> dict:
    """Return the corrected attribution refresh pack."""
    old_pack = build_trial2_hyperfine_attribution_split_audit_pack()
    first_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    corrected_pack = build_trial2_hyperfine_g2_correction_materialization_pack()

    alpha_constants = dict(first_pack["alpha_constants"])
    observed_1s2s = float(first_pack["summary"]["hydrogen_1s2s_observed_hz"])
    observed_hfs = float(corrected_pack["summary"]["observed_hz"])
    g_over_2 = float(corrected_pack["summary"]["g_over_2"])

    alpha_eff_1s2s = invert_alpha_squared_surface(observed_1s2s)
    alpha_eff_hfs_corrected = invert_alpha_quartic_corrected_surface(observed_hfs, g_over_2=g_over_2)
    corrected_split = float((alpha_eff_hfs_corrected - alpha_eff_1s2s) / alpha_eff_1s2s)
    old_split = float(old_pack["summary"]["effective_alpha_split_relative"])

    gap_rows = [
        build_gap_row(
            alpha_label=label,
            alpha_value=float(alpha_constants[label]),
            alpha_eff_1s2s=alpha_eff_1s2s,
            alpha_eff_hfs=alpha_eff_hfs_corrected,
        )
        for label in ALPHA_LABEL_ORDER
    ]
    closest_to_1s2s = min(gap_rows, key=lambda row: float(row["abs_relative_gap_vs_alpha_eff_1s2s"]))
    closest_to_hfs = min(gap_rows, key=lambda row: float(row["abs_relative_gap_vs_alpha_eff_hfs"]))
    split_reduction_factor = float(abs(old_split) / abs(corrected_split)) if corrected_split != 0.0 else math.inf
    both_codata = (
        str(closest_to_1s2s["alpha_label"]) == "alpha_CODATA"
        and str(closest_to_hfs["alpha_label"]) == "alpha_CODATA"
    )

    return {
        "surface_effective_alphas": {
            "hydrogen_1s2s_gross_structure_baseline": alpha_eff_1s2s,
            "hydrogen_hyperfine_21cm_g2_corrected_baseline": alpha_eff_hfs_corrected,
        },
        "checkpoint_gap_rows": gap_rows,
        "summary": {
            "alpha_eff_1s2s": alpha_eff_1s2s,
            "alpha_eff_hfs_corrected": alpha_eff_hfs_corrected,
            "effective_alpha_split_relative_old": old_split,
            "effective_alpha_split_relative_corrected": corrected_split,
            "split_reduction_factor": split_reduction_factor,
            "closest_to_1s2s_alpha_label": str(closest_to_1s2s["alpha_label"]),
            "closest_to_hfs_corrected_alpha_label": str(closest_to_hfs["alpha_label"]),
            "both_surfaces_closest_to_codata_now": both_codata,
            "corrected_hyperfine_split_reduced_now": abs(corrected_split) < abs(old_split),
            "corrected_attribution_reading": (
                "the g/2-corrected hyperfine surface moves the implied alpha "
                "toward the same CODATA-side neighborhood already read by 1S-2S"
            ),
        },
        "trial2_hyperfine_corrected_attribution_refresh_completed_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the corrected attribution backend directly."""
    pack = build_trial2_hyperfine_corrected_attribution_refresh_pack()
    summary = pack["summary"]
    print("[trial2_hyperfine_corrected_attribution_refresh_backend]")
    print(f"  alpha_eff_1s2s = {summary['alpha_eff_1s2s']}")
    print(f"  alpha_eff_hfs_corrected = {summary['alpha_eff_hfs_corrected']}")
    print(
        "  effective_alpha_split_relative_corrected = "
        f"{summary['effective_alpha_split_relative_corrected']}"
    )
    print(f"  both_surfaces_closest_to_codata_now = {summary['both_surfaces_closest_to_codata_now']}")


if __name__ == "__main__":
    main()
