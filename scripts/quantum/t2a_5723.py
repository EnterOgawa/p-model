#!/usr/bin/env python3
"""Generate 8.7.56.5723-.5726 patched-tail weighted-integral gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5719-5722",
        "updated_pack_trial2_beta_sensitivity_patched_tail_weighted_integral_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5723-5726"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "patched-tail weighted-integral gate / pure-analytic refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_patched_tail_weighted_integral_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_weighted_integral_sign_support_"
    "audited_tail_remainder_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_weighted_integral_sign_support_"
    "completed_tail_remainder_bound_followup_primary_conditional_reopen_secondary_next"
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


# 関数: gate で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the patched-tail weighted-integral gate."""
    return {
        "gate_a": "Gate A = patched-tail weighted-integral sign support is available now",
        "gate_b": "Gate B = patched-tail tail remainder is nonreversing now",
        "gate_c": "Gate C = patched-tail large-cutoff stability is available now",
    }


# 関数: `.5723-.5726` を実行する。

def main() -> None:
    """Execute the patched-tail weighted-integral gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now"
        ]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now"
        ]
    )
    gate_c = bool(
        gate_b
        and prior_summary["exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now"]
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_patched_tail_weighted_integral_sign_support_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 patched-tail weighted-integral sign support available now",
            sign_base.truth(gate_a),
            "The route refresh starts only once the full-domain patched dI_n/dbeta are fixed as negative on all tested h and asymptotic cutoffs.",
        ),
        sign_base.row(
            "gate_b_trial2_patched_tail_remainder_nonreversing_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 patched-tail tail remainder nonreversing now",
            sign_base.truth(gate_b),
            "The patched positive tail is allowed to contribute with mixed-sign u_beta so long as its net remainder cannot reverse the full weighted-integral sign.",
        ),
        sign_base.row(
            "gate_c_trial2_patched_tail_cutoff_stable_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 patched-tail cutoff stable now",
            sign_base.truth(gate_c),
            "Large-cutoff stability is required so that the patched-tail sign support does not collapse when the asymptotic continuation is pushed farther out.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "tail_match_x": float(prior_summary["tail_match_x"]),
        "d_i2_tail_fraction_max": float(prior_summary["d_i2_tail_fraction_max"]),
        "d_i2_tail_cutoff_rel_spread": float(
            prior_summary["d_i2_tail_cutoff_rel_spread"]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_weighted_integral_sign_support_available_now": bool(
            gate_a
        ),
        "exact_trial2_beta_sensitivity_patched_tail_remainder_nonreversing_now": bool(
            gate_b
        ),
        "exact_trial2_beta_sensitivity_patched_tail_cutoff_stable_now": bool(gate_c),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": False,
        "updated_pack_trial2_patched_tail_remainder_bound_followup_required_now": bool(
            gate_c
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_patched_tail_remainder_bound_followup"
            if gate_c
            else None
        ),
        "recommended_next_route_or_none": "8.7.56.5727-5730" if gate_c else None,
        "no_unconditional_next_official_branch_now": False if gate_c else True,
    }

    payload = {
        "step_tag": STEP_TAG,
        "step_name": STEP_NAME,
        "summary": summary,
        "rows": rows,
        "formulae": build_formulae(),
        "notes": {
            "gate_meaning": (
                "Patched-tail sign support is now official. The next honest blocker is one analytic remainder bound for the admissible positive-decay continuation."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
