#!/usr/bin/env python3
"""Fix the honest verdict after attribution split and third-surface refresh.

Purpose:
    Once the hyperfine win is localized and the third-surface inventory is
    refreshed, Trial-2 needs one honest gate deciding whether the current
    observable-comparison table can be promoted from watch to pass. This backend
    keeps the verdict mechanical.

Inputs:
    - scripts/quantum/trial2_first_multi_observable_comparison_refresh_backend.py
    - scripts/quantum/trial2_hyperfine_attribution_split_audit_backend.py
    - scripts/quantum/trial2_third_independent_surface_inventory_refresh_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5955-.5958` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_first_multi_observable_comparison_refresh_backend import (
    build_trial2_first_multi_observable_comparison_refresh_pack,
)
from scripts.quantum.trial2_hyperfine_attribution_split_audit_backend import (
    build_trial2_hyperfine_attribution_split_audit_pack,
)
from scripts.quantum.trial2_third_independent_surface_inventory_refresh_backend import (
    build_trial2_third_independent_surface_inventory_refresh_pack,
)


# 関数: `.5955-.5958` 用の watch/pass gate pack を返す。
def build_trial2_multi_observable_watch_pass_gate_pack() -> dict:
    """Return the retained watch/pass gate pack for the current table."""
    multi_pack = build_trial2_first_multi_observable_comparison_refresh_pack()
    attribution_pack = build_trial2_hyperfine_attribution_split_audit_pack()
    third_pack = build_trial2_third_independent_surface_inventory_refresh_pack()

    split_watch = bool(multi_pack["summary"]["split_watch_verdict_now"])
    attribution_localized = bool(attribution_pack["summary"]["hyperfine_attribution_split_localized_now"])
    genuine_third_available = bool(third_pack["summary"]["genuine_third_independent_surface_available_now"])
    pass_available = split_watch and attribution_localized and genuine_third_available
    watch_retained = split_watch and attribution_localized and not genuine_third_available

    return {
        "summary": {
            "split_watch_verdict_now": split_watch,
            "hyperfine_attribution_split_localized_now": attribution_localized,
            "genuine_third_independent_surface_available_now": genuine_third_available,
            "multi_observable_pass_available_now": bool(pass_available),
            "multi_observable_watch_retained_now": bool(watch_retained),
            "current_honest_reading": (
                "the split itself is now localized, but a third genuinely new "
                "independent alpha-explicit surface is still missing"
            ),
        },
        "trial2_multi_observable_pass_available_now": bool(pass_available),
        "trial2_multi_observable_watch_retained_now": bool(watch_retained),
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the multi-observable watch/pass gate backend directly."""
    pack = build_trial2_multi_observable_watch_pass_gate_pack()
    summary = pack["summary"]
    print("[trial2_multi_observable_watch_pass_gate_backend]")
    print(f"  split_watch_verdict_now = {summary['split_watch_verdict_now']}")
    print(
        "  genuine_third_independent_surface_available_now = "
        f"{summary['genuine_third_independent_surface_available_now']}"
    )
    print(
        "  multi_observable_watch_retained_now = "
        f"{summary['multi_observable_watch_retained_now']}"
    )


if __name__ == "__main__":
    main()
