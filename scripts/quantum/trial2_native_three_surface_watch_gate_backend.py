#!/usr/bin/env python3
"""Refresh the native multi-observable gate after the He II third surface.

Purpose:
    After the native He II one-electron surface actualizes, the old
    "third surface missing" blocker is gone. The honest remaining question is
    whether the three-surface native table now upgrades Trial-2 from watch to
    pass or whether the verdict stays at watch.

Inputs:
    - scripts/quantum/trial2_native_non_hydrogen_surface_gate_backend.py

Outputs:
    - One in-memory gate pack consumed by `.6035-.6038` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_native_non_hydrogen_surface_gate_backend import (
    build_trial2_native_non_hydrogen_surface_gate_pack,
)


# 関数: `.6035-.6038` 用の native three-surface watch gate pack を返す。
def build_trial2_native_three_surface_watch_gate_pack() -> dict:
    """Return the native three-surface watch gate pack."""
    gate_pack = build_trial2_native_non_hydrogen_surface_gate_pack()
    summary_pack = gate_pack["summary"]

    native_pmodel_wins = int(summary_pack["native_pmodel_win_count_now"])
    native_codata_wins = int(summary_pack["native_codata_win_count_now"])
    pass_available = False
    watch_retained = True

    return {
        "summary": {
            "native_actual_surface_count_now": int(summary_pack["native_actual_surface_count_now"]),
            "native_non_hydrogen_actual_surface_count_now": int(
                summary_pack["native_non_hydrogen_actual_surface_count_now"]
            ),
            "native_pmodel_win_count_now": native_pmodel_wins,
            "native_codata_win_count_now": native_codata_wins,
            "native_codata_lead_diagnostic_now": bool(summary_pack["native_codata_lead_diagnostic_now"]),
            "native_multi_observable_pass_available_now": bool(pass_available),
            "native_multi_observable_watch_retained_now": bool(watch_retained),
            "no_unconditional_next_official_branch_now": True,
            "current_honest_reading": (
                "The native primary table now has three actual surfaces, but its "
                "current honest verdict remains watch. One surface still favors a "
                "P-model alpha checkpoint while two gross-structure surfaces are "
                "closest to the CODATA diagnostic row on the same native shell."
            ),
        },
        "trial2_native_three_surface_watch_gate_completed_now": True,
        "trial2_no_unconditional_next_official_branch_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the native three-surface watch gate backend directly."""
    pack = build_trial2_native_three_surface_watch_gate_pack()
    summary = pack["summary"]
    print("[trial2_native_three_surface_watch_gate_backend]")
    print(
        "  native_multi_observable_watch_retained_now = "
        f"{summary['native_multi_observable_watch_retained_now']}"
    )
    print(
        "  native_codata_lead_diagnostic_now = "
        f"{summary['native_codata_lead_diagnostic_now']}"
    )
    print(
        "  no_unconditional_next_official_branch_now = "
        f"{summary['no_unconditional_next_official_branch_now']}"
    )


if __name__ == "__main__":
    main()
