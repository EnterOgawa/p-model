#!/usr/bin/env python3
"""Generate 8.7.56.5719-.5722 patched-tail weighted-integral audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_patched_tail_weighted_integral_followup_backend import (
    build_trial2_beta_sensitivity_patched_tail_weighted_integral_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5715-5718",
        "updated_pack_trial2_beta_sensitivity_admissible_tail_patch_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "96_trial2_numeric_alpha_vector_qball_patched_tail_weighted_integral_followup_audit.md"
)

STEP_TAG = "8.7.56.5719-5722"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "patched-tail weighted-integral sign-support audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_patched_tail_weighted_integral_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_pure_analytic_refinement_reopened_raw_tail_artifact_detected_"
    "positive_decay_tail_patch_primary_conditional_reopen_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_weighted_integral_sign_support_"
    "audited_tail_remainder_gate_next"
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
    """Return whether the patched-tail note carries the expected claims."""
    patterns = (
        "patched tail",
        "weighted integral",
        "nonreversing",
        "tail remainder",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the patched-tail weighted-integral audit."""
    return {
        "patched_tail": (
            "y_tail^(patch)(x; beta) = y_beta(x_match) * (x_match/x) * exp(-kappa_beta * (x-x_match))"
        ),
        "u_beta_patch": (
            "u_beta^(patch)(x) = (y_(beta+h)^(patch)(x) - y_(beta-h)^(patch)(x)) / (2h)"
        ),
        "weighted_integral": (
            "dI_n/dbeta = n * integral y_beta^(n-1)(x) * u_beta^(patch)(x) * x^2 dx"
        ),
    }


# 関数: `.5719-.5722` を実行する。

def main() -> None:
    """Execute the patched-tail weighted-integral sign-support audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_patched_tail_weighted_integral_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)

    rows = [
        sign_base.row(
            "updated_pack_trial2_patched_tail_weighted_integral_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 patched-tail weighted-integral route selected now",
            sign_base.truth(route_selected),
            "This branch starts only after the raw-tail artifact and positive-decay tail patch formula have already been fixed officially.",
        ),
        sign_base.row(
            "exact_trial2_patched_tail_weighted_integral_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 patched-tail weighted-integral note available now",
            sign_base.truth(note_available),
            "The note must explicitly state that the patched tail is judged by weighted-integral sign retention and nonreversing tail remainder, not by naive one-sign u_beta tail support.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_profile_available_now",
            "pass"
            if pack["exact_trial2_beta_sensitivity_patched_tail_profile_available_now"]
            else "reject",
            "exact Trial-2 beta-sensitivity patched-tail profile available now",
            sign_base.truth(
                pack["exact_trial2_beta_sensitivity_patched_tail_profile_available_now"]
            ),
            "Pass means the value-matched positive-decay continuation keeps the full patched profile positive on all tested cutoffs.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now",
            "pass"
            if pack[
                "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now"
            ]
            else "reject",
            "exact Trial-2 beta-sensitivity patched-tail weighted-integral sign support available now",
            sign_base.truth(
                pack[
                    "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now"
                ]
            ),
            "Pass means the full-domain patched dI_n/dbeta remain negative for n = 2, 3, 4 on all tested h and asymptotic cutoffs.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now",
            "pass"
            if pack[
                "exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now"
            ]
            else "reject",
            "exact Trial-2 beta-sensitivity patched-tail remainder nonreversing now",
            sign_base.truth(
                pack[
                    "exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now"
                ]
            ),
            "Pass means the patched positive tail contributes too little to overturn the already negative weighted integrals.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now",
            "pass"
            if pack["exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now"]
            else "reject",
            "exact Trial-2 beta-sensitivity patched-tail cutoff stable now",
            sign_base.truth(
                pack["exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now"]
            ),
            "Pass means the patched weighted-integral signs are no longer a short-cutoff accident once x_max is pushed into the asymptotic regime.",
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
            "This audit is allowed to keep the final pure analytic theorem open; it only has to remove the patched-tail sign-support blocker honestly.",
        ),
        sign_base.row(
            "updated_pack_trial2_patched_tail_remainder_bound_followup_required_now",
            "pass"
            if pack[
                "updated_pack_trial2_patched_tail_remainder_bound_followup_required_now"
            ]
            else "reject",
            "updated-pack Trial-2 patched-tail remainder-bound followup required now",
            sign_base.truth(
                pack[
                    "updated_pack_trial2_patched_tail_remainder_bound_followup_required_now"
                ]
            ),
            "Once patched weighted-integral sign support is fixed, the next honest blocker is no longer the raw tail itself but one analytic remainder bound for the admissible tail continuation.",
        ),
    ]

    order2 = pack["order_summaries"]["2"]
    order3 = pack["order_summaries"]["3"]
    order4 = pack["order_summaries"]["4"]
    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "tail_match_x": float(pack["tail_match_x"]),
        "u_beta_tail_positive_fraction_min": float(
            pack["u_beta_tail_positive_fraction_min"]
        ),
        "u_beta_tail_positive_fraction_max": float(
            pack["u_beta_tail_positive_fraction_max"]
        ),
        "d_i2_total_min": float(order2["weighted_total_integral_min"]),
        "d_i2_total_max": float(order2["weighted_total_integral_max"]),
        "d_i2_tail_fraction_max": float(order2["tail_remainder_abs_fraction_max"]),
        "d_i2_tail_cutoff_rel_spread": float(order2["tail_cutoff_rel_spread"]),
        "d_i3_tail_fraction_max": float(order3["tail_remainder_abs_fraction_max"]),
        "d_i4_tail_fraction_max": float(order4["tail_remainder_abs_fraction_max"]),
        "exact_trial2_beta_sensitivity_patched_tail_profile_available_now": bool(
            pack["exact_trial2_beta_sensitivity_patched_tail_profile_available_now"]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now": bool(
            pack[
                "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now"
            ]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now": bool(
            pack["exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now"]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now": bool(
            pack["exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now"]
        ),
        "updated_pack_trial2_patched_tail_remainder_bound_followup_required_now": bool(
            pack["updated_pack_trial2_patched_tail_remainder_bound_followup_required_now"]
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
                "The patched positive-decay tail no longer needs one-sign u_beta support. "
                "It only has to preserve the signs of the full weighted beta-derivative integrals "
                "while keeping the tail remainder nonreversing."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
