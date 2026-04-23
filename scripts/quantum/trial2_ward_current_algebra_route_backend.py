#!/usr/bin/env python3
"""
Audit the Trial-2 Ward identity / current-algebra route on the current pack.

This backend does not replay the exhausted overlap/Jost/scattering/coupling
branches. It asks one narrower question:

    does the selected-extension-native pack actually materialize one
    conserved-current / Ward / current-algebra surface that reads out alpha
    independently of the blind scalar form-factor pack?

The current pack already carries two older theorem-side facts:

    - the conserved background Noether current J_Noether^mu[Q]
    - the same-field source no-go for J_eff^mu[a;Q]

The route only survives if those older facts are promoted into one
selected-extension-native Ward/current-algebra surface rather than collapsing
back to charge normalization F(0)=1.
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

PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
CHARGE_CURRENT_AUDIT = PUBLIC_OUT / (
    "q_8_7_56_2591_2594_updated_pack_exact_charge_current_derivation_audit_"
    "declaration_gate_metrics.json"
)
SOURCE_THEOREM_AUDIT = PUBLIC_OUT / (
    "q_8_7_56_2607_2610_updated_pack_exact_source_theorem_closeout_audit_"
    "declaration_gate_metrics.json"
)


# 関数: 必須 input artifact の存在を確認する。
def require(path: Path) -> None:
    """Abort immediately when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON artifact into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


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


# 関数: selected-extension pack 上の current/Ward/algebra surface 有無を診断する。

def build_trial2_ward_current_algebra_pack(
    ell_values: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """Build one current-pack diagnostic pack for the Ward/current-algebra route."""
    require(CHARGE_CURRENT_AUDIT)
    require(SOURCE_THEOREM_AUDIT)

    charge_summary = read_json(CHARGE_CURRENT_AUDIT)["summary"]
    source_summary = read_json(SOURCE_THEOREM_AUDIT)["summary"]
    recompute_pack = build_selected_extension_solver_recompute_pack(
        ell_values=ell_values
    )
    qext_pack = build_selected_extension_solver_side_extra_q_range_numeric_rerun_pack(
        ell_values=ell_values
    )

    flattened_key_paths = flatten_key_paths(recompute_pack) | flatten_key_paths(qext_pack)
    selected_extension_current_hits = collect_token_hits(
        flattened_key_paths,
        (
            "current_vertex",
            "charge_current",
            "noether",
            "j_eff",
            "j_noether",
        ),
    )
    selected_extension_ward_hits = collect_token_hits(
        flattened_key_paths,
        (
            "ward",
            "continuity",
            "q_mu",
            "divergence",
            "conserved_current",
        ),
    )
    selected_extension_current_algebra_hits = collect_token_hits(
        flattened_key_paths,
        (
            "current_algebra",
            "commutator",
            "equal_time",
            "charge_operator",
            "generator",
            "jacobi",
        ),
    )

    soft_form_factor_zero = float(recompute_pack["F_blind_recomp_pack"]["zero"])
    soft_alpha_naive = compute_alpha_from_form_factor(soft_form_factor_zero)
    alpha_target = float(qext_pack["alpha_target"])
    soft_alpha_target_ratio = float(soft_alpha_naive / alpha_target)
    soft_alpha_target_relative_mismatch = float(
        abs(soft_alpha_naive - alpha_target) / alpha_target
    )

    background_noether_current_available_now = bool(
        charge_summary["updated_pack_exact_charge_current_noether_closure_available_now"]
    )
    same_field_source_no_go_available_now = bool(
        source_summary["updated_pack_exact_source_theorem_no_go_verdict_passed"]
        and source_summary["updated_pack_same_field_source_zero_fixed"]
    )
    selected_extension_scalar_surface_available_now = bool(
        set(("zero", "q_theory_over_m0", "m0"))
        <= set(recompute_pack["F_blind_recomp_pack"].keys())
        and bool(qext_pack["selected_extension_solver_side_extra_q_range_numeric_rerun_available_now"])
    )
    selected_extension_current_surface_available_now = bool(
        selected_extension_current_hits
    )
    selected_extension_ward_identity_surface_available_now = bool(
        selected_extension_ward_hits
    )
    selected_extension_current_algebra_surface_available_now = bool(
        selected_extension_current_algebra_hits
    )
    naive_soft_limit_charge_normalization_collapse_now = bool(
        math.isclose(soft_form_factor_zero, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        and soft_alpha_target_relative_mismatch > 9.0
    )
    independent_ward_alpha_readout_available_now = bool(
        selected_extension_current_surface_available_now
        and selected_extension_ward_identity_surface_available_now
        and selected_extension_current_algebra_surface_available_now
        and not naive_soft_limit_charge_normalization_collapse_now
    )
    ward_current_algebra_negative_closeout_available_now = bool(
        background_noether_current_available_now
        and same_field_source_no_go_available_now
        and selected_extension_scalar_surface_available_now
        and not selected_extension_current_surface_available_now
        and not selected_extension_ward_identity_surface_available_now
        and not selected_extension_current_algebra_surface_available_now
        and naive_soft_limit_charge_normalization_collapse_now
        and not independent_ward_alpha_readout_available_now
    )
    conditional_reopen_refresh_required_now = bool(
        ward_current_algebra_negative_closeout_available_now
    )

    return {
        "selected_extension_label": recompute_pack["selected_extension_label"],
        "soft_form_factor_zero": soft_form_factor_zero,
        "soft_alpha_naive": soft_alpha_naive,
        "alpha_target": alpha_target,
        "soft_alpha_target_ratio": soft_alpha_target_ratio,
        "soft_alpha_target_relative_mismatch": soft_alpha_target_relative_mismatch,
        "background_noether_current_available_now": (
            background_noether_current_available_now
        ),
        "same_field_source_no_go_available_now": (
            same_field_source_no_go_available_now
        ),
        "selected_extension_scalar_surface_available_now": (
            selected_extension_scalar_surface_available_now
        ),
        "selected_extension_current_hits": selected_extension_current_hits,
        "selected_extension_ward_hits": selected_extension_ward_hits,
        "selected_extension_current_algebra_hits": (
            selected_extension_current_algebra_hits
        ),
        "selected_extension_current_surface_available_now": (
            selected_extension_current_surface_available_now
        ),
        "selected_extension_ward_identity_surface_available_now": (
            selected_extension_ward_identity_surface_available_now
        ),
        "selected_extension_current_algebra_surface_available_now": (
            selected_extension_current_algebra_surface_available_now
        ),
        "naive_soft_limit_charge_normalization_collapse_now": (
            naive_soft_limit_charge_normalization_collapse_now
        ),
        "independent_ward_alpha_readout_available_now": (
            independent_ward_alpha_readout_available_now
        ),
        "ward_current_algebra_negative_closeout_available_now": (
            ward_current_algebra_negative_closeout_available_now
        ),
        "conditional_reopen_refresh_required_now": (
            conditional_reopen_refresh_required_now
        ),
    }


# 関数: helper 実行時に compact summary を返す。

def main() -> None:
    """Run the Ward/current-algebra helper and print a compact summary."""
    pack = build_trial2_ward_current_algebra_pack()
    summary = {
        "selected_extension_label": pack["selected_extension_label"],
        "soft_alpha_naive": pack["soft_alpha_naive"],
        "alpha_target": pack["alpha_target"],
        "background_noether_current_available_now": pack[
            "background_noether_current_available_now"
        ],
        "same_field_source_no_go_available_now": pack[
            "same_field_source_no_go_available_now"
        ],
        "selected_extension_current_surface_available_now": pack[
            "selected_extension_current_surface_available_now"
        ],
        "selected_extension_ward_identity_surface_available_now": pack[
            "selected_extension_ward_identity_surface_available_now"
        ],
        "selected_extension_current_algebra_surface_available_now": pack[
            "selected_extension_current_algebra_surface_available_now"
        ],
        "ward_current_algebra_negative_closeout_available_now": pack[
            "ward_current_algebra_negative_closeout_available_now"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper summary を出力する。

if __name__ == "__main__":
    main()
