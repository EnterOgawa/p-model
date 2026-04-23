#!/usr/bin/env python3
"""Cut the second independent-observable rerun gate for Trial-2.

Purpose:
    After the first actual independent rerun becomes available on Hydrogen
    1S-2S gross structure, the next honest gate is whether a second
    independent, alpha-explicit, rerun-ready surface exists inside the current
    public pack.

    This backend keeps the gate narrow:

    1. Hydrogen 1S-2S remains the first actual surface,
    2. Lamb absolute formula is checked via its own materialization audit,
    3. weak beta-decay explicit-alpha route remains a reserve candidate only.

Inputs:
    - scripts/quantum/trial2_first_actual_independent_observable_rerun_gate_backend.py
    - scripts/quantum/trial2_lamb_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5927-.5930` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_first_actual_independent_observable_rerun_gate_backend import (
    build_trial2_first_actual_independent_rerun_gate_pack,
)
from scripts.quantum.trial2_lamb_absolute_alpha_formula_materialization_backend import (
    build_trial2_lamb_absolute_alpha_formula_pack,
)
from scripts.quantum.trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend import (
    build_trial2_weak_beta_decay_explicit_alpha_formula_pack,
)


# 関数: `.5927-.5930` 用の second-rerun gate pack を返す。
def build_trial2_second_independent_rerun_gate_pack() -> dict:
    """Return the retained second independent-observable rerun gate pack."""
    first_pack = build_trial2_first_actual_independent_rerun_gate_pack()
    lamb_pack = build_trial2_lamb_absolute_alpha_formula_pack()
    weak_pack = build_trial2_weak_beta_decay_explicit_alpha_formula_pack()

    selected_first = first_pack["selected_observable"]
    current_actual_surface_count = 1 if bool(
        first_pack["summary"]["first_actual_independent_observable_rerun_available_now"]
    ) else 0
    second_ready = False

    return {
        "first_gate_summary": first_pack["summary"],
        "lamb_pack_summary": lamb_pack["summary"],
        "weak_pack_summary": weak_pack["summary"],
        "retained_only_surface": selected_first,
        "prediction_table": first_pack["prediction_table"],
        "summary": {
            "second_independent_observable_rerun_available_now": second_ready,
            "current_actual_surface_count_now": int(current_actual_surface_count),
            "retained_only_surface_id": str(selected_first["surface_id"]),
            "retained_only_surface_label": str(selected_first["label"]),
            "lamb_absolute_formula_ready_now": bool(
                lamb_pack["summary"]["lamb_absolute_formula_materialized_now"]
            ),
            "weak_explicit_formula_ready_now": bool(
                weak_pack["trial2_weak_beta_decay_primary_ready_now"]
            ),
        },
        "trial2_second_independent_observable_rerun_available_now": second_ready,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the second independent-observable rerun gate directly."""
    pack = build_trial2_second_independent_rerun_gate_pack()
    summary = pack["summary"]
    print("[trial2_second_independent_observable_rerun_gate_backend]")
    print(
        "  second_independent_observable_rerun_available_now = "
        f"{summary['second_independent_observable_rerun_available_now']}"
    )
    print(
        "  current_actual_surface_count_now = "
        f"{summary['current_actual_surface_count_now']}"
    )
    print(f"  retained_only_surface_id = {summary['retained_only_surface_id']}")


if __name__ == "__main__":
    main()
