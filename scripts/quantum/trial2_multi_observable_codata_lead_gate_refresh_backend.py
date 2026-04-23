#!/usr/bin/env python3
"""Refresh the multi-observable gate after the third independent surface.

Purpose:
    Once a third genuinely independent alpha-explicit surface is available,
    Trial-2 needs one honest aggregate gate asking whether the current
    multi-observable table has moved beyond the two-surface CODATA-lead watch.

Inputs:
    - scripts/quantum/trial2_third_independent_surface_gate_second_refresh_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5979-.5982` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_third_independent_surface_gate_second_refresh_backend import (
    build_trial2_third_independent_surface_gate_second_refresh_pack,
)


# 関数: `.5979-.5982` 用の aggregate gate pack を返す。
def build_trial2_multi_observable_codata_lead_gate_refresh_pack() -> dict:
    """Return the refreshed three-surface aggregate gate pack."""
    gate_pack = build_trial2_third_independent_surface_gate_second_refresh_pack()
    surface_rows = list(gate_pack["surface_rows"])
    pmodel_win_count = sum(1 for row in surface_rows if bool(row["pmodel_wins_now"]))
    codata_win_count = sum(1 for row in surface_rows if bool(row["codata_wins_now"]))
    codata_sweep = codata_win_count == len(surface_rows) and pmodel_win_count == 0

    return {
        "surface_rows": surface_rows,
        "summary": {
            "current_actual_surface_count_now": int(len(surface_rows)),
            "surface_ids_now": [str(row["surface_id"]) for row in surface_rows],
            "pmodel_win_count_now": int(pmodel_win_count),
            "codata_win_count_now": int(codata_win_count),
            "codata_sweep_verdict_now": bool(codata_sweep),
            "multi_observable_pass_available_now": False,
            "multi_observable_watch_retained_now": True,
            "multi_observable_codata_lead_watch_retained_now": bool(codata_sweep),
            "current_honest_reading": (
                "The current three-surface Hydrogen-only table leans to CODATA on "
                "all actual surfaces, but it still remains a watch rather than a "
                "final pass/reject because the comparison is not yet cross-sector "
                "and the deterministic surfaces are baseline-level."
            ),
        },
        "trial2_multi_observable_codata_lead_gate_refresh_completed_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the refreshed multi-observable CODATA-lead gate backend directly."""
    pack = build_trial2_multi_observable_codata_lead_gate_refresh_pack()
    summary = pack["summary"]
    print("[trial2_multi_observable_codata_lead_gate_refresh_backend]")
    print(f"  current_actual_surface_count_now = {summary['current_actual_surface_count_now']}")
    print(f"  pmodel_win_count_now = {summary['pmodel_win_count_now']}")
    print(f"  codata_win_count_now = {summary['codata_win_count_now']}")
    print(f"  codata_sweep_verdict_now = {summary['codata_sweep_verdict_now']}")


if __name__ == "__main__":
    main()
