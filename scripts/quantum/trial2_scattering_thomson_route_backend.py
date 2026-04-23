#!/usr/bin/env python3
"""
Audit the Trial-2 scattering amplitude / Thomson-limit route on the current pack.

This backend does not replay the exhausted overlap/Jost/coupling routes. It asks
one narrower question:

    does the selected-extension-native pack actually materialize one independent
    low-energy scattering amplitude / Thomson-limit readout for alpha?

The audit uses only already materialized selected-extension surfaces:

    - O_recomp_sel^(pilot-HS)
    - helper-backed extra-q surface

If the current pack exposes only scalar form-factor checkpoints and the naive
soft limit collapses to charge normalization F(0)=1, the scattering / Thomson
route closes negatively and Ward / current algebra becomes the next honest
primary route.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.selected_extension_solver_recompute_backend import (
    build_selected_extension_solver_recompute_pack,
)
from scripts.quantum.selected_extension_solver_side_extra_q_range_numeric_rerun_backend import (
    build_selected_extension_solver_side_extra_q_range_numeric_rerun_pack,
)


# 関数: blind form factor から scalar alpha proxy を計算する。
def compute_alpha_from_form_factor(form_factor_value: float) -> float:
    """Return alpha = F(q)^2 / (4 pi) for one scalar form-factor value."""
    return float(form_factor_value * form_factor_value / (4.0 * math.pi))


# 関数: dict/list の nested key path を平坦化する。

def flatten_key_paths(obj: object, prefix: str = "") -> set[str]:
    """Return one flat set of nested key paths for summary-level surface scans."""
    key_paths: set[str] = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            key_paths.add(next_prefix)
            key_paths |= flatten_key_paths(value, next_prefix)

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            next_prefix = f"{prefix}[{index}]"
            key_paths.add(next_prefix)
            key_paths |= flatten_key_paths(value, next_prefix)

    return key_paths


# 関数: flat key path 集合から token hit を返す。

def collect_token_hits(key_paths: set[str], tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return sorted key paths containing any token from the given token set."""
    lowered = tuple(token.lower() for token in tokens)
    hits = [
        path
        for path in sorted(key_paths)
        if any(token in path.lower() for token in lowered)
    ]
    return tuple(hits)


# 関数: scattering / Thomson route audit pack を構成する。

def build_trial2_scattering_thomson_pack(
    ell_values: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Build one current-pack diagnostic pack for the scattering / Thomson route."""
    recompute_pack = build_selected_extension_solver_recompute_pack(
        ell_values=ell_values
    )
    qext_pack = build_selected_extension_solver_side_extra_q_range_numeric_rerun_pack(
        ell_values=ell_values
    )

    flattened_key_paths = flatten_key_paths(recompute_pack) | flatten_key_paths(qext_pack)
    low_energy_surface_hits = collect_token_hits(
        flattened_key_paths,
        ("omega", "frequency", "soft_photon", "soft_limit"),
    )
    angular_surface_hits = collect_token_hits(
        flattened_key_paths,
        ("angle", "theta", "angular"),
    )
    polarization_surface_hits = collect_token_hits(
        flattened_key_paths,
        ("helicity", "lambda_in", "lambda_out", "pol_in", "pol_out"),
    )
    thomson_surface_hits = collect_token_hits(
        flattened_key_paths,
        ("thomson", "cross_section", "sigma_t"),
    )

    soft_form_factor_zero = float(recompute_pack["F_blind_recomp_pack"]["zero"])
    soft_alpha_naive = compute_alpha_from_form_factor(soft_form_factor_zero)
    alpha_target = float(qext_pack["alpha_target"])
    soft_alpha_target_ratio = float(soft_alpha_naive / alpha_target)
    soft_alpha_target_relative_mismatch = float(
        abs(soft_alpha_naive - alpha_target) / alpha_target
    )

    q_theory_over_m0 = float(recompute_pack["retained_q_window"]["q_theory_over_m0"])
    q_theory_form_factor = float(recompute_pack["F_blind_recomp_pack"]["q_theory_over_m0"])
    q_theory_alpha = float(recompute_pack["alpha_blind_recomp_at_q_theory"])

    best_extra_label_vs_alpha_target = str(qext_pack["best_extra_label_vs_alpha_target"])
    best_extra_q_diagnostic = qext_pack["q_surface_diagnostics"][
        best_extra_label_vs_alpha_target
    ]
    best_extra_q_over_m0 = float(best_extra_q_diagnostic["q_over_m0"])
    best_extra_alpha = float(best_extra_q_diagnostic["alpha_blind_value"])
    best_extra_alpha_target_residual = float(
        qext_pack["best_extra_alpha_target_residual"]
    )
    best_extra_legacy_phase3_sideband = bool(
        best_extra_q_diagnostic["legacy_phase3_sideband"]
    )

    low_energy_surface_available_now = bool(low_energy_surface_hits)
    angular_surface_available_now = bool(angular_surface_hits)
    polarization_surface_available_now = bool(polarization_surface_hits)
    thomson_surface_available_now = bool(thomson_surface_hits)

    naive_soft_limit_charge_normalization_collapse_now = bool(
        math.isclose(soft_form_factor_zero, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        and soft_alpha_target_relative_mismatch > 9.0
    )
    legacy_phase3_sideband_target_proximity_only_now = bool(
        qext_pack["phase3_sideband_carryover_only_now"]
        and best_extra_legacy_phase3_sideband
        and best_extra_label_vs_alpha_target == "best_global_signed_q"
    )
    independent_scattering_surface_available_now = bool(
        low_energy_surface_available_now
        and angular_surface_available_now
        and polarization_surface_available_now
        and thomson_surface_available_now
    )

    return {
        "selected_extension_label": recompute_pack["selected_extension_label"],
        "retained_q_window": recompute_pack["retained_q_window"],
        "soft_form_factor_zero": soft_form_factor_zero,
        "soft_alpha_naive": soft_alpha_naive,
        "alpha_target": alpha_target,
        "soft_alpha_target_ratio": soft_alpha_target_ratio,
        "soft_alpha_target_relative_mismatch": soft_alpha_target_relative_mismatch,
        "q_theory_over_m0": q_theory_over_m0,
        "q_theory_form_factor": q_theory_form_factor,
        "q_theory_alpha": q_theory_alpha,
        "best_extra_label_vs_alpha_target": best_extra_label_vs_alpha_target,
        "best_extra_q_over_m0": best_extra_q_over_m0,
        "best_extra_alpha": best_extra_alpha,
        "best_extra_alpha_target_residual": best_extra_alpha_target_residual,
        "best_extra_legacy_phase3_sideband": best_extra_legacy_phase3_sideband,
        "low_energy_surface_hits": low_energy_surface_hits,
        "angular_surface_hits": angular_surface_hits,
        "polarization_surface_hits": polarization_surface_hits,
        "thomson_surface_hits": thomson_surface_hits,
        "low_energy_surface_available_now": low_energy_surface_available_now,
        "angular_surface_available_now": angular_surface_available_now,
        "polarization_surface_available_now": polarization_surface_available_now,
        "thomson_surface_available_now": thomson_surface_available_now,
        "naive_soft_limit_charge_normalization_collapse_now": (
            naive_soft_limit_charge_normalization_collapse_now
        ),
        "legacy_phase3_sideband_target_proximity_only_now": (
            legacy_phase3_sideband_target_proximity_only_now
        ),
        "independent_scattering_surface_available_now": (
            independent_scattering_surface_available_now
        ),
    }


# 関数: helper 実行時に compact summary を返す。

def main() -> None:
    """Run the scattering / Thomson audit helper and print a compact summary."""
    pack = build_trial2_scattering_thomson_pack()
    summary = {
        "selected_extension_label": pack["selected_extension_label"],
        "soft_alpha_naive": pack["soft_alpha_naive"],
        "alpha_target": pack["alpha_target"],
        "soft_alpha_target_relative_mismatch": pack[
            "soft_alpha_target_relative_mismatch"
        ],
        "best_extra_label_vs_alpha_target": pack["best_extra_label_vs_alpha_target"],
        "best_extra_alpha_target_residual": pack["best_extra_alpha_target_residual"],
        "independent_scattering_surface_available_now": pack[
            "independent_scattering_surface_available_now"
        ],
        "naive_soft_limit_charge_normalization_collapse_now": pack[
            "naive_soft_limit_charge_normalization_collapse_now"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper summary を出力する。

if __name__ == "__main__":
    main()
