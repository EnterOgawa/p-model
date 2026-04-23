#!/usr/bin/env python3
"""Refresh whether any non-Hydrogen alpha-explicit rerun surface now exists.

Purpose:
    After the three-surface Hydrogen-only table leans to CODATA on all actual
    surfaces, the next honest reopen condition is a genuinely new non-Hydrogen
    independent alpha-explicit surface. This backend aggregates the retained
    Helium, Lamb, and weak-sector routes and fixes whether any such surface is
    actual in the current pack.

Inputs:
    - scripts/quantum/trial2_multi_observable_codata_lead_gate_refresh_backend.py
    - scripts/quantum/trial2_helium_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_lamb_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5987-.5990` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_helium_absolute_alpha_formula_materialization_backend import (
    build_trial2_helium_absolute_alpha_formula_materialization_pack,
)
from scripts.quantum.trial2_lamb_absolute_alpha_formula_materialization_backend import (
    build_trial2_lamb_absolute_alpha_formula_pack,
)
from scripts.quantum.trial2_multi_observable_codata_lead_gate_refresh_backend import (
    build_trial2_multi_observable_codata_lead_gate_refresh_pack,
)
from scripts.quantum.trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend import (
    build_trial2_weak_beta_decay_explicit_alpha_formula_pack,
)


# 関数: `.5987-.5990` 用の non-Hydrogen gate pack を返す。
def build_trial2_non_hydrogen_surface_gate_refresh_pack() -> dict:
    """Return the refreshed gate for non-Hydrogen alpha-explicit surfaces."""
    multi_pack = build_trial2_multi_observable_codata_lead_gate_refresh_pack()
    helium_pack = build_trial2_helium_absolute_alpha_formula_materialization_pack()
    lamb_pack = build_trial2_lamb_absolute_alpha_formula_pack()
    weak_pack = build_trial2_weak_beta_decay_explicit_alpha_formula_pack()

    candidate_rows = [
        {
            "candidate_id": "helium_absolute_formula",
            "absolute_formula_available_now": bool(
                helium_pack["summary"]["helium_absolute_formula_available_now"]
            ),
            "rerun_ready_surface_available_now": bool(
                helium_pack["summary"]["helium_rerun_ready_surface_available_now"]
            ),
        },
        {
            "candidate_id": "lamb_absolute_formula",
            "absolute_formula_available_now": bool(
                lamb_pack["summary"]["lamb_absolute_formula_materialized_now"]
            ),
            "rerun_ready_surface_available_now": bool(
                lamb_pack["summary"]["lamb_absolute_formula_materialized_now"]
            ),
        },
        {
            "candidate_id": "weak_explicit_formula",
            "absolute_formula_available_now": bool(
                weak_pack["summary"]["weak_explicit_formula_ready_count"] > 0
            ),
            "rerun_ready_surface_available_now": bool(
                weak_pack["summary"]["weak_explicit_formula_ready_count"] > 0
            ),
        },
    ]
    non_hydrogen_actual_count = sum(
        1 for row in candidate_rows if bool(row["rerun_ready_surface_available_now"])
    )

    return {
        "candidate_rows": candidate_rows,
        "summary": {
            "hydrogen_actual_surface_count_now": int(
                multi_pack["summary"]["current_actual_surface_count_now"]
            ),
            "hydrogen_surface_ids_now": list(multi_pack["summary"]["surface_ids_now"]),
            "non_hydrogen_candidate_route_count_now": int(len(candidate_rows)),
            "non_hydrogen_actual_surface_count_now": int(non_hydrogen_actual_count),
            "non_hydrogen_surface_available_now": bool(non_hydrogen_actual_count > 0),
            "hydrogen_only_table_retained_now": bool(non_hydrogen_actual_count == 0),
            "current_honest_reading": (
                "The retained actual table is still Hydrogen-only. Helium, Lamb, and "
                "weak routes are all audited but none currently materialize one "
                "non-Hydrogen alpha-explicit rerun-ready surface."
            ),
        },
        "trial2_non_hydrogen_surface_available_now": bool(non_hydrogen_actual_count > 0),
        "trial2_non_hydrogen_surface_unavailable_now": bool(non_hydrogen_actual_count == 0),
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the non-Hydrogen surface gate refresh backend directly."""
    pack = build_trial2_non_hydrogen_surface_gate_refresh_pack()
    summary = pack["summary"]
    print("[trial2_non_hydrogen_surface_gate_refresh_backend]")
    print(
        "  hydrogen_actual_surface_count_now = "
        f"{summary['hydrogen_actual_surface_count_now']}"
    )
    print(
        "  non_hydrogen_actual_surface_count_now = "
        f"{summary['non_hydrogen_actual_surface_count_now']}"
    )
    print(
        "  hydrogen_only_table_retained_now = "
        f"{summary['hydrogen_only_table_retained_now']}"
    )


if __name__ == "__main__":
    main()
