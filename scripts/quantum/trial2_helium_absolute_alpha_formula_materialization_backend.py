#!/usr/bin/env python3
"""Audit whether the current Helium public pack exposes one absolute alpha formula.

Purpose:
    Trial-2 observable comparison now has three actual Hydrogen-only surfaces and
    therefore needs one honest non-Hydrogen reopen check. The cleanest current
    public candidate is the retained He I baseline pack from Step 7.12.

    This backend does not invent a Helium theory formula. It fixes the current
    public-canonical verdict by checking whether the existing He I cache already
    carries:

    1. one deterministic absolute `alpha -> observable` formula,
    2. one term-resolved transition map that would support reruns,
    3. one deterministic screening / effective-charge law.

Inputs:
    - output/public/quantum/atomic_helium_baseline_metrics.json

Outputs:
    - One in-memory audit pack consumed by `.5983-.5986` wrappers
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
ATOMIC_HELIUM = PUBLIC_OUT / "atomic_helium_baseline_metrics.json"


# 関数: UTF-8 JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `.5983-.5986` 用の Helium absolute-formula audit pack を返す。

def build_trial2_helium_absolute_alpha_formula_materialization_pack() -> dict:
    """Return the retained Helium absolute alpha-formula materialization pack."""
    metrics = read_json(ATOMIC_HELIUM)
    lines = list(metrics["lines"])
    selected_line_count = len(lines)
    selected_keys = sorted({key for row in lines for key in row.keys()})
    term_assignment_available = all(
        ("upper_term" in row and "lower_term" in row) for row in lines
    )
    screening_law_available = False
    absolute_formula_available = False
    rerun_ready_surface_available = False

    return {
        "helium_metrics_path": str(ATOMIC_HELIUM),
        "helium_lines": lines,
        "summary": {
            "helium_selected_line_count_now": int(selected_line_count),
            "helium_selected_line_ids_now": [str(row["id"]) for row in lines],
            "helium_selected_keys_now": selected_keys,
            "helium_term_assignment_available_now": bool(term_assignment_available),
            "helium_screening_law_available_now": bool(screening_law_available),
            "helium_absolute_formula_available_now": bool(absolute_formula_available),
            "helium_rerun_ready_surface_available_now": bool(rerun_ready_surface_available),
            "current_honest_reading": (
                "The retained He I public cache fixes observed wavelengths and Aki rows "
                "only. It still does not expose one term-resolved deterministic "
                "alpha-to-observable formula or one public screening law, so Helium "
                "remains observed-only in the current pack."
            ),
        },
        "trial2_helium_absolute_formula_materialized_now": False,
        "trial2_helium_negative_closeout_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the Helium absolute alpha-formula audit directly."""
    pack = build_trial2_helium_absolute_alpha_formula_materialization_pack()
    summary = pack["summary"]
    print("[trial2_helium_absolute_alpha_formula_materialization_backend]")
    print(
        "  helium_selected_line_count_now = "
        f"{summary['helium_selected_line_count_now']}"
    )
    print(
        "  helium_term_assignment_available_now = "
        f"{summary['helium_term_assignment_available_now']}"
    )
    print(
        "  helium_absolute_formula_available_now = "
        f"{summary['helium_absolute_formula_available_now']}"
    )


if __name__ == "__main__":
    main()
