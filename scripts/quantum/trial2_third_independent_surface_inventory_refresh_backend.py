#!/usr/bin/env python3
"""Refresh the inventory for a genuinely new third alpha-explicit surface.

Purpose:
    After the first two-surface comparison turns into a split watch, the next
    honest question is whether the current public pack already contains one
    genuinely new third independent alpha-explicit rerun surface. This backend
    separates replay-capable Hydrogen gross-structure lines from genuinely new
    families and fixes whether a third independent surface actually exists.

Inputs:
    - output/public/quantum/atomic_hydrogen_baseline_metrics.json
    - output/public/quantum/atomic_helium_baseline_metrics.json
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_lamb_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend.py

Outputs:
    - One in-memory inventory pack consumed by `.5951-.5954` wrappers
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend import (
    ALPHA_CODATA,
    ALPHA_P_4D_CAN,
    ALPHA_P_4D_VERTEX,
)
from scripts.quantum.trial2_lamb_absolute_alpha_formula_materialization_backend import (
    build_trial2_lamb_absolute_alpha_formula_pack,
)
from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    ALPHA_COMMON,
    ALPHA_P_FROZEN,
    C_M_PER_S,
    H_J_S,
    M_E_KG,
    M_P_KG,
)
from scripts.quantum.trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend import (
    build_trial2_weak_beta_decay_explicit_alpha_formula_pack,
)


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
ATOMIC_HYDROGEN = PUBLIC_OUT / "atomic_hydrogen_baseline_metrics.json"
ATOMIC_HELIUM = PUBLIC_OUT / "atomic_helium_baseline_metrics.json"

ALPHA_ROWS = {
    "alpha_P_frozen": ALPHA_P_FROZEN,
    "alpha_common": ALPHA_COMMON,
    "alpha_P_4D_can": ALPHA_P_4D_CAN,
    "alpha_P_4D_vertex": ALPHA_P_4D_VERTEX,
    "alpha_CODATA": ALPHA_CODATA,
}

HYDROGEN_GROSS_DELTAS = {
    "H_I_Lyα": {"n_hi": 2, "n_lo": 1, "delta": 1.0 - 1.0 / 4.0},
    "H_I_Hα": {"n_hi": 3, "n_lo": 2, "delta": 1.0 / 4.0 - 1.0 / 9.0},
    "H_I_Hβ": {"n_hi": 4, "n_lo": 2, "delta": 1.0 / 4.0 - 1.0 / 16.0},
    "H_I_Hγ": {"n_hi": 5, "n_lo": 2, "delta": 1.0 / 4.0 - 1.0 / 25.0},
}


# 関数: JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: hydrogen gross-structure family の predicted frequency を返す。

def hydrogen_gross_frequency_hz(*, alpha_value: float, delta_n: float) -> float:
    """Return one Hydrogen gross-structure line frequency under the Coulomb baseline."""
    mu_red = (M_E_KG * M_P_KG) / (M_E_KG + M_P_KG)
    return float((mu_red * (C_M_PER_S**2) * (alpha_value**2) / (2.0 * H_J_S)) * delta_n)


# 関数: 1 本の hydrogen replay row を作る。

def build_hydrogen_replay_row(*, line_id: str, observed_hz: float, delta_n: float) -> dict:
    """Return one replay-capable Hydrogen gross-structure line row."""
    predictions = []

    for alpha_label, alpha_value in ALPHA_ROWS.items():
        predicted_hz = hydrogen_gross_frequency_hz(alpha_value=alpha_value, delta_n=delta_n)
        rel_error = float((predicted_hz - observed_hz) / observed_hz)
        predictions.append(
            {
                "alpha_label": alpha_label,
                "alpha_value": float(alpha_value),
                "predicted_hz": predicted_hz,
                "relative_error_vs_observed": rel_error,
            }
        )

    predictions_sorted = sorted(predictions, key=lambda row: abs(float(row["relative_error_vs_observed"])))
    alpha_eff = float(math.sqrt(observed_hz / hydrogen_gross_frequency_hz(alpha_value=1.0, delta_n=delta_n)))
    best = predictions_sorted[0]
    return {
        "surface_id": line_id,
        "family_id": "hydrogen_gross_structure_replay_family",
        "formula": "nu(alpha) = (mu_red * c^2 * alpha^2 / (2 h)) * (1/n_lo^2 - 1/n_hi^2)",
        "n_hi": int(HYDROGEN_GROSS_DELTAS[line_id]["n_hi"]),
        "n_lo": int(HYDROGEN_GROSS_DELTAS[line_id]["n_lo"]),
        "delta_n": float(delta_n),
        "observed_hz": float(observed_hz),
        "alpha_eff": alpha_eff,
        "best_overall_alpha_label": str(best["alpha_label"]),
        "best_overall_relative_error_vs_observed": float(best["relative_error_vs_observed"]),
        "predictions": predictions_sorted,
        "same_family_as_1s2s_now": True,
        "genuinely_new_independent_surface_now": False,
    }


# 関数: `.5951-.5954` 用の inventory refresh pack を返す。

def build_trial2_third_independent_surface_inventory_refresh_pack() -> dict:
    """Return the refreshed third-surface inventory pack."""
    hydrogen = read_json(ATOMIC_HYDROGEN)
    helium = read_json(ATOMIC_HELIUM)
    lamb_pack = build_trial2_lamb_absolute_alpha_formula_pack()
    weak_pack = build_trial2_weak_beta_decay_explicit_alpha_formula_pack()

    replay_rows = []

    for line in hydrogen["lines"]:
        line_id = str(line["id"])
        if line_id not in HYDROGEN_GROSS_DELTAS:
            continue

        replay_rows.append(
            build_hydrogen_replay_row(
                line_id=line_id,
                observed_hz=float(line["frequency_THz"]) * 1.0e12,
                delta_n=float(HYDROGEN_GROSS_DELTAS[line_id]["delta"]),
            )
        )

    hydrogen_replay_all_codata_best = all(
        str(row["best_overall_alpha_label"]) == "alpha_CODATA" for row in replay_rows
    )
    helium_surface_available = False
    return {
        "hydrogen_replay_rows": replay_rows,
        "helium_observed_only_rows": list(helium["lines"]),
        "summary": {
            "hydrogen_gross_replay_candidate_count_now": int(len(replay_rows)),
            "hydrogen_gross_replay_all_codata_best_now": bool(hydrogen_replay_all_codata_best),
            "hydrogen_gross_replay_is_genuinely_new_now": False,
            "helium_absolute_formula_available_now": bool(helium_surface_available),
            "lamb_absolute_formula_available_now": bool(
                lamb_pack["summary"]["lamb_absolute_formula_materialized_now"]
            ),
            "weak_explicit_formula_available_now": bool(
                weak_pack["summary"]["weak_explicit_formula_ready_count"] > 0
            ),
            "genuine_third_independent_surface_available_now": False,
            "current_honest_reading": (
                "Hydrogen Balmer/Lyman lines are replay-capable but remain in the "
                "same gross-structure alpha^2 family as 1S-2S, while Helium, Lamb, "
                "and weak-sector absolute formulas are still unavailable."
            ),
        },
        "trial2_genuine_third_independent_surface_available_now": False,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the third-surface inventory refresh backend directly."""
    pack = build_trial2_third_independent_surface_inventory_refresh_pack()
    summary = pack["summary"]
    print("[trial2_third_independent_surface_inventory_refresh_backend]")
    print(
        "  hydrogen_gross_replay_candidate_count_now = "
        f"{summary['hydrogen_gross_replay_candidate_count_now']}"
    )
    print(
        "  hydrogen_gross_replay_all_codata_best_now = "
        f"{summary['hydrogen_gross_replay_all_codata_best_now']}"
    )
    print(
        "  genuine_third_independent_surface_available_now = "
        f"{summary['genuine_third_independent_surface_available_now']}"
    )


if __name__ == "__main__":
    main()
