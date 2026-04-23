#!/usr/bin/env python3
"""Refresh the pure-continuum closure wording after the patched-tail promotion.

Purpose:
    `.5707-.5710` already fixed one official verdict:

        first-principles direct-alpha closure is completed,
        while pure analytic operator-level continuum refinement is deferred.

    `.5727-.5734` then added one new theorem-quality surface that did not exist
    when the above verdict was frozen:

        the admissible patched tail now admits an explicit closed-form
        remainder bound on [X, +inf), so the weighted-integral route supports
        one honest pure-continuum promotion.

    This backend does not replay any solve. It only asks whether the old final
    wording should now be refreshed to reflect the new state honestly:

    1. direct-alpha closure remains completed,
    2. one patched-tail pure-continuum closure layer is now synchronized into
       the v2 theorem wording, and
    3. full operator-level continuum refinement is still open and therefore
       remains deferred to v3.0.

Inputs:
    - `.5707-.5710` final-closure declaration artifact
    - `.5731-.5734` patched-tail remainder-bound declaration artifact

Outputs:
    - One in-memory refresh pack consumed by `.5735-.5742`

Assumptions:
    - No new parameter is introduced
    - No new heavy replay is performed
    - The refreshed wording hardens the continuum statement only at the
      patched weighted-integral level, not at the full operator level
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
FINAL_CLOSURE_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5707-5710",
        "updated_pack_trial2_beta_sensitivity_final_closure_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PATCHED_REMAINDER_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5731-5734",
        "updated_pack_trial2_beta_sensitivity_patched_tail_remainder_bound_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]


# 関数: declaration artifact の summary を返す。
def read_summary(path: Path) -> dict:
    """Return the summary object from one declaration artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["summary"])


# 関数: patched-tail pure-continuum closure refresh pack を返す。

def build_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_refresh_pack() -> dict:
    """Return one refresh pack for the patched-tail pure-continuum wording update."""
    final_summary = read_summary(FINAL_CLOSURE_GATE)
    patched_summary = read_summary(PATCHED_REMAINDER_GATE)

    first_principles_direct_alpha_closure_completed_now = bool(
        final_summary["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
    )
    patched_tail_pure_continuum_promotion_available_now = bool(
        patched_summary[
            "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now"
        ]
    )
    operator_level_continuum_refinement_available_now = bool(
        final_summary["exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"]
    )

    patched_tail_pure_continuum_closure_completed_now = bool(
        first_principles_direct_alpha_closure_completed_now
        and patched_tail_pure_continuum_promotion_available_now
    )
    v2_theorem_wording_upgrade_available_now = bool(
        patched_tail_pure_continuum_closure_completed_now
        and not operator_level_continuum_refinement_available_now
    )
    pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now = bool(
        patched_tail_pure_continuum_closure_completed_now
        and not operator_level_continuum_refinement_available_now
    )
    updated_pack_trial2_patched_tail_pure_continuum_closure_gate_required_now = bool(
        v2_theorem_wording_upgrade_available_now
    )

    return {
        "beta_common_root": float(final_summary["beta_common_root"]),
        "alpha_common_value": float(final_summary["alpha_common_value"]),
        "alpha_common_rel_error_vs_target": float(
            final_summary["alpha_common_rel_error_vs_target"]
        ),
        "tail_match_x": float(patched_summary["tail_match_x"]),
        "x_cutoff": float(patched_summary["x_cutoff"]),
        "analytic_remainder_bound_n2": float(
            patched_summary["analytic_remainder_bound_n2"]
        ),
        "analytic_remainder_over_total_abs_min_n2": float(
            patched_summary["analytic_remainder_over_total_abs_min_n2"]
        ),
        "exact_trial2_first_principles_direct_alpha_closure_completed_now": bool(
            first_principles_direct_alpha_closure_completed_now
        ),
        "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now": bool(
            patched_tail_pure_continuum_promotion_available_now
        ),
        "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now": bool(
            patched_tail_pure_continuum_closure_completed_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": bool(
            operator_level_continuum_refinement_available_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now": bool(
            pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now
        ),
        "exact_trial2_v2_theorem_wording_upgrade_available_now": bool(
            v2_theorem_wording_upgrade_available_now
        ),
        "updated_pack_trial2_patched_tail_pure_continuum_closure_gate_required_now": bool(
            updated_pack_trial2_patched_tail_pure_continuum_closure_gate_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the patched-tail pure-continuum closure refresh backend directly."""
    pack = (
        build_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_refresh_pack()
    )
    print("[trial2-beta-patched-tail-pure-continuum-closure-refresh]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"alpha_common_value = {pack['alpha_common_value']:.16f}")
    print(
        "patched_tail_pure_continuum_closure_completed = "
        f"{pack['exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now']}"
    )
    print(
        "operator_level_continuum_refinement_deferred_to_v3 = "
        f"{pack['exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now']}"
    )


if __name__ == "__main__":
    main()
