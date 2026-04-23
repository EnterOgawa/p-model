#!/usr/bin/env python3
"""Refresh the post-He II native watch gate after the same-family audit.

Purpose:
    Once the first He II surface actualizes, the next honest question is whether
    the retained He II hydrogenic family still contains a stronger replay route.
    If not, the live blockers contract to the relativistic bridge and genuine
    precision-correction routes only.

Inputs:
    - scripts/quantum/trial2_native_helium_ion_family_strength_audit_backend.py
    - scripts/quantum/trial2_native_three_surface_watch_gate_backend.py

Outputs:
    - One in-memory gate pack consumed by `.6043-.6046` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_native_helium_ion_family_strength_audit_backend import (
    build_trial2_native_helium_ion_family_strength_audit_pack,
)
from scripts.quantum.trial2_native_three_surface_watch_gate_backend import (
    build_trial2_native_three_surface_watch_gate_pack,
)


# 関数: `.6043-.6046` 用の post-He II watch gate pack を返す。
def build_trial2_native_post_heii_watch_gate_pack() -> dict:
    """Return the post-He II native watch gate pack."""
    family_pack = build_trial2_native_helium_ion_family_strength_audit_pack()
    watch_pack = build_trial2_native_three_surface_watch_gate_pack()
    family_summary = family_pack["summary"]
    watch_summary = watch_pack["summary"]

    return {
        "summary": {
            "native_actual_surface_count_now": int(watch_summary["native_actual_surface_count_now"]),
            "native_pmodel_win_count_now": int(watch_summary["native_pmodel_win_count_now"]),
            "native_codata_win_count_now": int(watch_summary["native_codata_win_count_now"]),
            "native_multi_observable_watch_retained_now": bool(
                watch_summary["native_multi_observable_watch_retained_now"]
            ),
            "heii_family_line_count_now": int(family_summary["heii_family_line_count_now"]),
            "heii_family_strongest_pmodel_line_id_now": str(
                family_summary["heii_family_strongest_pmodel_line_id_now"]
            ),
            "heii_family_stronger_than_46867_route_available_now": bool(
                family_summary["heii_family_stronger_than_46867_route_available_now"]
            ),
            "no_unconditional_next_official_branch_now": True,
            "current_honest_reading": (
                "The retained He II hydrogenic family does not supply a stronger "
                "same-family replay than the already selected 468.67 nm surface. "
                "The live blockers are therefore one native relativistic Halpha bridge "
                "or one genuinely new precision/native-family extension."
            ),
        },
        "trial2_native_post_heii_watch_gate_completed_now": True,
        "trial2_no_unconditional_next_official_branch_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the post-He II native watch gate directly."""
    pack = build_trial2_native_post_heii_watch_gate_pack()
    summary = pack["summary"]
    print("[trial2_native_post_heii_watch_gate_backend]")
    print(
        "  heii_family_stronger_than_46867_route_available_now = "
        f"{summary['heii_family_stronger_than_46867_route_available_now']}"
    )
    print(
        "  native_multi_observable_watch_retained_now = "
        f"{summary['native_multi_observable_watch_retained_now']}"
    )
    print(
        "  no_unconditional_next_official_branch_now = "
        f"{summary['no_unconditional_next_official_branch_now']}"
    )


if __name__ == "__main__":
    main()
