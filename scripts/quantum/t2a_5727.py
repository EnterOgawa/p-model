#!/usr/bin/env python3
"""Generate 8.7.56.5727-.5730 patched-tail analytic remainder-bound artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_followup_backend import (
    build_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5723-5726",
        "updated_pack_trial2_beta_sensitivity_patched_tail_weighted_integral_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "97_trial2_numeric_alpha_vector_qball_patched_tail_analytic_remainder_bound_followup_audit.md"
)

STEP_TAG = "8.7.56.5727-5730"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "patched-tail analytic remainder-bound audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_patched_tail_remainder_bound_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_weighted_integral_sign_support_"
    "completed_tail_remainder_bound_followup_primary_conditional_reopen_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_analytic_remainder_bound_"
    "audited_pure_continuum_promotion_gate_next"
)


# 関数: JSON/CSV artifact を書き出す。
def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# 関数: note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the note carries the expected analytic-remainder claims."""
    patterns = (
        "analytic remainder bound",
        "largest retained cutoff",
        "pure-continuum promotion",
        "closed-form upper bound",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the analytic remainder-bound audit."""
    return {
        "patched_tail": (
            "y_tail^(patch)(x; beta) = y_beta(x_match) * (x_match/x) * exp(-kappa_beta * (x-x_match))"
        ),
        "u_tail": (
            "u_tail^(patch)(x; beta) = (partial_beta y_beta(x_match) + y_beta(x_match) * (beta/kappa_beta) * (x-x_match)) * (x_match/x) * exp(-kappa_beta * (x-x_match))"
        ),
        "remainder_bound": (
            "|R_n^(tail)(X)| <= n * Y_match^(n-1) * U_match * J_n,0^(ub)(X) + "
            "n * Y_match^n * (beta/kappa_beta) * J_n,1^(ub)(X)"
        ),
        "upper_integrals": (
            "J_n,0^(ub)(X) = x_match^n * X^(2-n) * exp(-n*kappa_beta*(X-x_match)) / (n*kappa_beta), "
            "J_n,1^(ub)(X) = x_match^n * X^(2-n) * exp(-n*kappa_beta*(X-x_match)) * "
            "((X-x_match)/(n*kappa_beta) + 1/(n^2*kappa_beta^2))"
        ),
    }


# 関数: `.5727-.5730` を実行する。

def main() -> None:
    """Execute the patched-tail analytic remainder-bound audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    prior_weighted_support = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now"
        ]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_patched_tail_remainder_bound_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 patched-tail remainder-bound route selected now",
            sign_base.truth(route_selected),
            "This audit starts only after the patched-tail weighted-integral gate is already official.",
        ),
        sign_base.row(
            "exact_trial2_patched_tail_remainder_bound_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 patched-tail remainder-bound note available now",
            sign_base.truth(note_available),
            "The note must state that the missing layer is one explicit analytic bound for the omitted tail beyond the largest tested cutoff.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now",
            "pass" if prior_weighted_support else "reject",
            "exact Trial-2 beta-sensitivity patched-tail weighted-integral sign support available now",
            sign_base.truth(prior_weighted_support),
            "The analytic tail bound is only meaningful once the cutoff-limited weighted integrals are already fixed as negative.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now",
            "pass"
            if pack[
                "exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now"
            ]
            else "reject",
            "exact Trial-2 beta-sensitivity patched-tail analytic remainder bound available now",
            sign_base.truth(
                pack[
                    "exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now"
                ]
            ),
            "Pass means the omitted patched tail beyond X = 140 admits one explicit closed-form upper bound that stays below the already-fixed negative sign margin for n = 2, 3, 4.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now",
            "pass"
            if pack[
                "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now"
            ]
            else "reject",
            "exact Trial-2 beta-sensitivity patched-tail pure-continuum promotion available now",
            sign_base.truth(
                pack[
                    "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now"
                ]
            ),
            "Pass means the patched weighted-integral sign support now survives the omitted tail analytically, so the route no longer depends on a finite cutoff accident.",
        ),
        sign_base.row(
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now",
            "pass"
            if pack[
                "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"
            ]
            else "reject",
            "exact Trial-2 pure analytic operator-level continuum refinement available now",
            sign_base.truth(
                pack[
                    "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"
                ]
            ),
            "This audit promotes the patched weighted-integral route, not yet the full operator-level continuum theorem.",
        ),
        sign_base.row(
            "updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now",
            "pass"
            if pack[
                "updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now"
            ]
            else "reject",
            "updated-pack Trial-2 patched-tail pure-continuum closure refresh required now",
            sign_base.truth(
                pack[
                    "updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now"
                ]
            ),
            "Once the omitted tail is bounded analytically, the next honest blocker is no longer the patched-tail remainder itself but how to fold this promotion back into the reopened pure-analytic closure wording.",
        ),
    ]

    order2 = next(row for row in pack["order_rows"] if row["order_n"] == 2)
    order3 = next(row for row in pack["order_rows"] if row["order_n"] == 3)
    order4 = next(row for row in pack["order_rows"] if row["order_n"] == 4)
    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "tail_match_x": float(pack["tail_match_x"]),
        "x_cutoff": float(pack["x_cutoff"]),
        "y_match_abs_max": float(pack["y_match_abs_max"]),
        "u_match_abs_max": float(pack["u_match_abs_max"]),
        "u_match_rel_spread": float(pack["u_match_rel_spread"]),
        "analytic_remainder_bound_n2": float(order2["remainder_abs_bound"]),
        "analytic_remainder_bound_n3": float(order3["remainder_abs_bound"]),
        "analytic_remainder_bound_n4": float(order4["remainder_abs_bound"]),
        "analytic_remainder_over_total_abs_min_n2": float(
            order2["bound_over_total_abs_min"]
        ),
        "analytic_remainder_over_total_abs_min_n3": float(
            order3["bound_over_total_abs_min"]
        ),
        "analytic_remainder_over_total_abs_min_n4": float(
            order4["bound_over_total_abs_min"]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now": bool(
            pack[
                "exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now": bool(
            pack[
                "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now"
            ]
        ),
        "updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now": bool(
            pack[
                "updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now"
            ]
        ),
    }

    payload = {
        "step_tag": STEP_TAG,
        "step_name": STEP_NAME,
        "summary": summary,
        "rows": rows,
        "formulae": build_formulae(),
        "notes": {
            "audit_meaning": (
                "This branch does not replay weighted-integral signs. It proves that the omitted patched tail beyond the largest retained cutoff is analytically too small to reverse the already-fixed sign."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
