#!/usr/bin/env python3
"""Cut the first multi-observable comparison gate for Trial-2.

Purpose:
    Observable-comparison mainline only becomes statistically meaningful once at
    least two independent, alpha-explicit, rerun-ready surfaces exist. This
    backend fixes whether the current public pack has crossed that threshold.

Inputs:
    - scripts/quantum/trial2_first_actual_independent_observable_rerun_gate_backend.py
    - scripts/quantum/trial2_second_independent_observable_rerun_gate_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5931-.5934` wrappers
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
from scripts.quantum.trial2_second_independent_observable_rerun_gate_backend import (
    build_trial2_second_independent_rerun_gate_pack,
)


# 関数: `.5931-.5934` 用の multi-observable gate pack を返す。
def build_trial2_first_multi_observable_comparison_gate_pack() -> dict:
    """Return the retained first multi-observable comparison gate pack."""
    first_pack = build_trial2_first_actual_independent_rerun_gate_pack()
    second_pack = build_trial2_second_independent_rerun_gate_pack()

    only_surface = first_pack["selected_observable"]
    multi_available = bool(second_pack["summary"]["second_independent_observable_rerun_available_now"])
    current_actual_surface_count = int(second_pack["summary"]["current_actual_surface_count_now"])

    return {
        "first_gate_summary": first_pack["summary"],
        "second_gate_summary": second_pack["summary"],
        "retained_only_surface": only_surface,
        "prediction_table": first_pack["prediction_table"],
        "summary": {
            "first_multi_observable_comparison_available_now": multi_available,
            "current_actual_surface_count_now": int(current_actual_surface_count),
            "minimum_surface_count_required_for_multi_now": 2,
            "missing_surface_count_now": int(max(0, 2 - current_actual_surface_count)),
            "retained_only_surface_id": str(only_surface["surface_id"]),
            "best_overall_alpha_label": str(first_pack["summary"]["best_overall_alpha_label"]),
            "best_pmodel_alpha_label": str(first_pack["summary"]["best_pmodel_alpha_label"]),
        },
        "trial2_first_multi_observable_comparison_available_now": multi_available,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the first multi-observable comparison gate directly."""
    pack = build_trial2_first_multi_observable_comparison_gate_pack()
    summary = pack["summary"]
    print("[trial2_first_multi_observable_comparison_gate_backend]")
    print(
        "  first_multi_observable_comparison_available_now = "
        f"{summary['first_multi_observable_comparison_available_now']}"
    )
    print(
        "  current_actual_surface_count_now = "
        f"{summary['current_actual_surface_count_now']}"
    )
    print(f"  missing_surface_count_now = {summary['missing_surface_count_now']}")


if __name__ == "__main__":
    main()
