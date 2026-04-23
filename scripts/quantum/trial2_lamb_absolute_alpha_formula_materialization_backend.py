#!/usr/bin/env python3
"""Audit whether the current public pack exposes one absolute Lamb alpha formula.

Purpose:
    Trial-2 observable comparison has one actual rerun surface already
    materialized on Hydrogen 1S-2S gross structure. The next honest question is
    whether the retained Lamb-shift side can also be promoted to an explicit
    absolute `alpha -> observable` formula inside the current public pack.

    This backend does not invent a new Lamb formula. It fixes the current
    public-canonical verdict by checking whether the existing QED-vacuum pack
    already carries:

    1. one deterministic absolute alpha formula,
    2. one rerun-ready Lamb observable surface,
    3. enough retained evidence to keep Lamb as a future candidate even when
       the absolute formula is still unavailable.

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - output/public/quantum/qed_vacuum_precision_metrics.json

Outputs:
    - One in-memory audit pack consumed by `.5923-.5926` wrappers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
)


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"


# 関数: 1 本の UTF-8 JSON payload を読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `.5923-.5926` 用の Lamb absolute-formula audit pack を返す。

def build_trial2_lamb_absolute_alpha_formula_pack() -> dict:
    """Return the retained Lamb absolute alpha-formula materialization pack."""
    qed_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    metrics = read_json(PUBLIC_OUT / "qed_vacuum_precision_metrics.json")

    hydrogen_surface = next(
        row
        for row in qed_pack["surfaces"]
        if str(row["surface_id"]) == "hydrogen_1s2s_gross_structure_baseline"
    )
    lamb_surface = next(
        row
        for row in qed_pack["surfaces"]
        if str(row["surface_id"]) == "lamb_shift_absolute_formula"
    )

    lamb_metrics = metrics["lamb_shift"]
    z_grid_count = len(list(lamb_metrics["z_grid"]))
    nuclear_table_count = len(list(lamb_metrics["nuclear_table4_mhz"]))
    hydrogen_is_only_actual = bool(hydrogen_surface["current_alpha_rerun_ready_now"]) and not bool(
        lamb_surface["current_alpha_rerun_ready_now"]
    )

    return {
        "qed_pack_summary": qed_pack["summary"],
        "lamb_surface": lamb_surface,
        "hydrogen_surface": hydrogen_surface,
        "summary": {
            "lamb_absolute_formula_materialized_now": bool(
                lamb_surface["current_alpha_rerun_ready_now"]
            ),
            "lamb_current_public_formula_available_now": bool(
                lamb_surface["formula"] is not None
            ),
            "lamb_structural_alpha_sensitivity_retained_now": (
                str(lamb_surface["alpha_dependency_kind"])
                == "structurally_alpha_sensitive_but_absolute_formula_unavailable"
            ),
            "hydrogen_surface_still_only_actual_rerun_surface_now": hydrogen_is_only_actual,
            "qed_actual_rerun_surface_count_now": int(
                sum(
                    1
                    for row in qed_pack["surfaces"]
                    if bool(row["current_alpha_rerun_ready_now"])
                    and bool(row["independent_observable_now"])
                )
            ),
            "lamb_z_grid_count": int(z_grid_count),
            "lamb_nuclear_table_count": int(nuclear_table_count),
        },
        "trial2_lamb_absolute_formula_materialized_now": False,
        "trial2_lamb_negative_closeout_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the Lamb absolute alpha-formula audit directly."""
    pack = build_trial2_lamb_absolute_alpha_formula_pack()
    summary = pack["summary"]
    print("[trial2_lamb_absolute_alpha_formula_materialization_backend]")
    print(
        "  lamb_absolute_formula_materialized_now = "
        f"{summary['lamb_absolute_formula_materialized_now']}"
    )
    print(
        "  hydrogen_surface_still_only_actual_rerun_surface_now = "
        f"{summary['hydrogen_surface_still_only_actual_rerun_surface_now']}"
    )
    print(f"  lamb_z_grid_count = {summary['lamb_z_grid_count']}")


if __name__ == "__main__":
    main()
