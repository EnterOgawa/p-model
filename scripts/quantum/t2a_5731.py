#!/usr/bin/env python3
"""Generate 8.7.56.5731-.5734 patched-tail remainder-bound gate artifacts."""

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
        "8.7.56.5727-5730",
        "updated_pack_trial2_beta_sensitivity_patched_tail_remainder_bound_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5731-5734"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "patched-tail remainder-bound gate / pure-analytic refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_patched_tail_remainder_bound_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_analytic_remainder_bound_"
    "audited_pure_continuum_promotion_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_analytic_remainder_bound_"
    "completed_pure_continuum_closure_refresh_primary_conditional_reopen_secondary_next"
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
    """Return formulas used by the patched-tail remainder-bound gate."""
    return {
        "gate_a": "Gate A = patched-tail analytic remainder bound is available now",
        "gate_b": "Gate B = patched-tail pure-continuum promotion is available now",
        "gate_c": "Gate C = the next honest blocker is a closure refresh, not another tail remainder replay",
    }


# 関数: `.5731-.5734` を実行する。

def main() -> None:
    """Execute the patched-tail remainder-bound gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now"
        ]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now"
        ]
    )
    gate_c = bool(
        gate_b
        and prior_summary[
            "updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now"
        ]
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_patched_tail_analytic_remainder_bound_available_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 patched-tail analytic remainder bound available now",
            sign_base.truth(gate_a),
            "The patched-tail refresh starts only once the omitted tail beyond X = 140 is analytically bounded below the retained sign margin.",
        ),
        sign_base.row(
            "gate_b_trial2_patched_tail_pure_continuum_promotion_available_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 patched-tail pure-continuum promotion available now",
            sign_base.truth(gate_b),
            "This gate closes the finite-cutoff loophole for the patched weighted-integral route: the sign support no longer depends on a truncated tail accident.",
        ),
        sign_base.row(
            "gate_c_trial2_patched_tail_pure_continuum_closure_refresh_required_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 patched-tail pure-continuum closure refresh required now",
            sign_base.truth(gate_c),
            "Once the omitted tail is controlled analytically, the next honest task is to fold that result back into the reopened pure-analytic refinement wording.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "tail_match_x": float(prior_summary["tail_match_x"]),
        "x_cutoff": float(prior_summary["x_cutoff"]),
        "analytic_remainder_bound_n2": float(
            prior_summary["analytic_remainder_bound_n2"]
        ),
        "analytic_remainder_over_total_abs_min_n2": float(
            prior_summary["analytic_remainder_over_total_abs_min_n2"]
        ),
        "exact_trial2_beta_sensitivity_patched_tail_analytic_remainder_bound_available_now": bool(
            gate_a
        ),
        "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_promotion_available_now": bool(
            gate_b
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": False,
        "updated_pack_trial2_patched_tail_pure_continuum_closure_refresh_required_now": bool(
            gate_c
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_patched_tail_pure_continuum_closure_refresh"
            if gate_c
            else None
        ),
        "recommended_next_route_or_none": "8.7.56.5735-5738" if gate_c else None,
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
                "Patched-tail analytic remainder control is now official. The next honest blocker is how to restate the reopened pure-analytic refinement after this continuum promotion, not another tail replay."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
