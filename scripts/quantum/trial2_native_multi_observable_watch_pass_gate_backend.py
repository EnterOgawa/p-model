#!/usr/bin/env python3
"""Fix the native multi-observable watch/pass gate after the Halpha audit.

Purpose:
    Trial-2 primary comparison now admits only P-model formula x P-model alpha.
    After cutting the native Halpha audit and the native third-surface gate, one
    final honest question remains: can the native Hydrogen table be promoted
    from split watch to pass, or does it remain at the two-surface split-watch
    stopping point?

Inputs:
    - scripts/quantum/trial2_native_third_surface_gate_backend.py

Outputs:
    - One in-memory gate pack consumed by `.6023-.6026` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_native_third_surface_gate_backend import (
    build_trial2_native_third_surface_gate_pack,
)


# 関数: `.6023-.6026` 用の native watch/pass gate pack を返す。
def build_trial2_native_multi_observable_watch_pass_gate_pack() -> dict:
    """Return the native multi-observable watch/pass gate pack."""
    third_gate_pack = build_trial2_native_third_surface_gate_pack()
    summary_pack = third_gate_pack["summary"]

    split_watch = bool(summary_pack["native_split_watch_retained_now"])
    third_available = bool(summary_pack["native_genuine_third_surface_available_now"])
    pass_available = split_watch and third_available
    watch_retained = split_watch and not third_available

    return {
        "summary": {
            "native_actual_surface_count_now": int(summary_pack["native_actual_surface_count_now"]),
            "native_pmodel_win_count_now": int(summary_pack["native_pmodel_win_count_now"]),
            "native_codata_win_count_now": int(summary_pack["native_codata_win_count_now"]),
            "native_split_watch_retained_now": split_watch,
            "native_genuine_third_surface_available_now": third_available,
            "native_multi_observable_pass_available_now": bool(pass_available),
            "native_multi_observable_watch_retained_now": bool(watch_retained),
            "no_unconditional_next_official_branch_now": True,
            "current_honest_reading": (
                "Trial-2 primary observable verdict remains a native two-surface "
                "split watch. No native relativistic third surface is available "
                "under the current public canon, so there is no honest promotion "
                "to native pass yet."
            ),
        },
        "trial2_native_multi_observable_watch_pass_gate_completed_now": True,
        "trial2_no_unconditional_next_official_branch_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the native multi-observable watch/pass gate backend directly."""
    pack = build_trial2_native_multi_observable_watch_pass_gate_pack()
    summary = pack["summary"]
    print("[trial2_native_multi_observable_watch_pass_gate_backend]")
    print(
        "  native_multi_observable_watch_retained_now = "
        f"{summary['native_multi_observable_watch_retained_now']}"
    )
    print(
        "  native_multi_observable_pass_available_now = "
        f"{summary['native_multi_observable_pass_available_now']}"
    )
    print(
        "  no_unconditional_next_official_branch_now = "
        f"{summary['no_unconditional_next_official_branch_now']}"
    )


if __name__ == "__main__":
    main()
