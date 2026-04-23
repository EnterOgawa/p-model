#!/usr/bin/env python3
"""Refresh the native watch gate after the local He I screening audit.

Purpose:
    The retained non-Hydrogen local pool currently consists of:

    1. one actual He II one-electron surface,
    2. same-family He II replays,
    3. one observed-only He I cache.

    Once the He II same-family replay is exhausted and the He I simple-screening
    route is cut negatively, the remaining reopen candidates contract to
    genuinely new bridges, precision corrections, or new non-Hydrogen families.

Inputs:
    - scripts/quantum/trial2_native_post_heii_watch_gate_backend.py
    - scripts/quantum/trial2_native_helium_simple_screening_audit_backend.py

Outputs:
    - One in-memory gate pack consumed by `.6051-.6054` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_native_helium_simple_screening_audit_backend import (
    build_trial2_native_helium_simple_screening_audit_pack,
)
from scripts.quantum.trial2_native_post_heii_watch_gate_backend import (
    build_trial2_native_post_heii_watch_gate_pack,
)


# 関数: `.6051-.6054` 用の local non-Hydrogen gate pack を返す。
def build_trial2_native_local_nonhydrogen_gate_pack() -> dict:
    """Return the local non-Hydrogen candidate gate pack."""
    post_heii_pack = build_trial2_native_post_heii_watch_gate_pack()
    helium_screening_pack = build_trial2_native_helium_simple_screening_audit_pack()
    post_heii_summary = post_heii_pack["summary"]
    helium_screening_summary = helium_screening_pack["summary"]

    stronger_local_non_hydrogen_route_available_now = bool(
        post_heii_summary["heii_family_stronger_than_46867_route_available_now"]
        or helium_screening_summary["helium_simple_screening_surface_ready_now"]
    )

    return {
        "summary": {
            "native_actual_surface_count_now": int(post_heii_summary["native_actual_surface_count_now"]),
            "native_pmodel_win_count_now": int(post_heii_summary["native_pmodel_win_count_now"]),
            "native_codata_win_count_now": int(post_heii_summary["native_codata_win_count_now"]),
            "native_multi_observable_watch_retained_now": bool(
                post_heii_summary["native_multi_observable_watch_retained_now"]
            ),
            "heii_family_stronger_than_46867_route_available_now": bool(
                post_heii_summary["heii_family_stronger_than_46867_route_available_now"]
            ),
            "helium_simple_screening_surface_ready_now": bool(
                helium_screening_summary["helium_simple_screening_surface_ready_now"]
            ),
            "stronger_local_non_hydrogen_route_available_now": bool(
                stronger_local_non_hydrogen_route_available_now
            ),
            "no_unconditional_next_official_branch_now": True,
            "current_honest_reading": (
                "The local non-Hydrogen pool is now exhausted inside the current pack: "
                "He II same-family replay is exhausted and the retained He I cache does "
                "not admit one honest constant-screening native surface. Further progress "
                "therefore needs a genuinely new relativistic bridge, a deterministic "
                "native precision correction, or a genuinely new non-Hydrogen family."
            ),
        },
        "trial2_native_local_nonhydrogen_gate_completed_now": True,
        "trial2_no_unconditional_next_official_branch_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the local non-Hydrogen gate directly."""
    pack = build_trial2_native_local_nonhydrogen_gate_pack()
    summary = pack["summary"]
    print("[trial2_native_local_nonhydrogen_gate_backend]")
    print(
        "  helium_simple_screening_surface_ready_now = "
        f"{summary['helium_simple_screening_surface_ready_now']}"
    )
    print(
        "  stronger_local_non_hydrogen_route_available_now = "
        f"{summary['stronger_local_non_hydrogen_route_available_now']}"
    )
    print(
        "  no_unconditional_next_official_branch_now = "
        f"{summary['no_unconditional_next_official_branch_now']}"
    )


if __name__ == "__main__":
    main()
