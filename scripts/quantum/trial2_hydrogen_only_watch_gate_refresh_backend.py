#!/usr/bin/env python3
"""Refresh the honest watch verdict after the non-Hydrogen gate is cut.

Purpose:
    The retained three-surface table already leans to CODATA on all actual
    surfaces. After auditing Helium, the remaining honest question is whether
    Trial-2 should still be described as a Hydrogen-only watch or whether some
    broader pass/reject verdict is now justified.

Inputs:
    - scripts/quantum/trial2_multi_observable_codata_lead_gate_refresh_backend.py
    - scripts/quantum/trial2_non_hydrogen_surface_gate_refresh_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5991-.5994` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_multi_observable_codata_lead_gate_refresh_backend import (
    build_trial2_multi_observable_codata_lead_gate_refresh_pack,
)
from scripts.quantum.trial2_non_hydrogen_surface_gate_refresh_backend import (
    build_trial2_non_hydrogen_surface_gate_refresh_pack,
)


# 関数: `.5991-.5994` 用の Hydrogen-only watch refresh pack を返す。
def build_trial2_hydrogen_only_watch_gate_refresh_pack() -> dict:
    """Return the refreshed Hydrogen-only watch gate pack."""
    multi_pack = build_trial2_multi_observable_codata_lead_gate_refresh_pack()
    non_hydrogen_pack = build_trial2_non_hydrogen_surface_gate_refresh_pack()

    codata_sweep = bool(multi_pack["summary"]["codata_sweep_verdict_now"])
    pass_unavailable = not bool(multi_pack["summary"]["multi_observable_pass_available_now"])
    non_hydrogen_unavailable = bool(
        non_hydrogen_pack["summary"]["non_hydrogen_actual_surface_count_now"] == 0
    )
    hydrogen_only_watch_retained = codata_sweep and pass_unavailable and non_hydrogen_unavailable

    return {
        "summary": {
            "hydrogen_actual_surface_count_now": int(
                multi_pack["summary"]["current_actual_surface_count_now"]
            ),
            "pmodel_win_count_now": int(multi_pack["summary"]["pmodel_win_count_now"]),
            "codata_win_count_now": int(multi_pack["summary"]["codata_win_count_now"]),
            "codata_sweep_verdict_now": bool(codata_sweep),
            "non_hydrogen_actual_surface_count_now": int(
                non_hydrogen_pack["summary"]["non_hydrogen_actual_surface_count_now"]
            ),
            "multi_observable_pass_available_now": False,
            "hydrogen_only_watch_retained_now": bool(hydrogen_only_watch_retained),
            "no_unconditional_next_official_branch_now": True,
            "current_honest_reading": (
                "Trial-2 remains a Hydrogen-only CODATA-lead watch. The current pack "
                "has three actual Hydrogen surfaces and zero actual non-Hydrogen "
                "alpha-explicit rerun surfaces, so no final pass/reject upgrade is "
                "honest yet."
            ),
        },
        "trial2_hydrogen_only_watch_retained_now": bool(hydrogen_only_watch_retained),
        "trial2_no_unconditional_next_official_branch_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the Hydrogen-only watch gate refresh backend directly."""
    pack = build_trial2_hydrogen_only_watch_gate_refresh_pack()
    summary = pack["summary"]
    print("[trial2_hydrogen_only_watch_gate_refresh_backend]")
    print(
        "  codata_sweep_verdict_now = "
        f"{summary['codata_sweep_verdict_now']}"
    )
    print(
        "  non_hydrogen_actual_surface_count_now = "
        f"{summary['non_hydrogen_actual_surface_count_now']}"
    )
    print(
        "  hydrogen_only_watch_retained_now = "
        f"{summary['hydrogen_only_watch_retained_now']}"
    )


if __name__ == "__main__":
    main()
