#!/usr/bin/env python3
"""Generate 8.7.56.5739-.5742 patched-tail pure-continuum closure gate artifacts."""

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
        "8.7.56.5735-5738",
        "updated_pack_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_refresh_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5739-5742"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "patched-tail pure-continuum closure gate / v2 theorem wording sync"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_admissible_positive_decay_tail_patch_pure_continuum_closure_"
    "audited_v2_wording_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_principles_direct_alpha_closure_completed_patched_tail_"
    "pure_continuum_closure_synced_operator_level_refinement_deferred_v3_"
    "conditional_reopen_only_next"
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
    """Return formulas used by the patched-tail pure-continuum closure gate."""
    return {
        "gate_a": "Gate A = first-principles direct-alpha closure remains completed now",
        "gate_b": "Gate B = patched-tail pure-continuum closure is synchronized into the v2 theorem wording now",
        "gate_c": "Gate C = full operator-level continuum refinement remains deferred to v3 and there is no unconditional next official branch now",
    }


# 関数: `.5739-.5742` を実行する。

def main() -> None:
    """Execute the patched-tail pure-continuum closure gate / wording sync."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now"
        ]
        and prior_summary["exact_trial2_v2_theorem_wording_upgrade_available_now"]
    )
    gate_c = bool(
        gate_b
        and prior_summary[
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now"
        ]
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_first_principles_direct_alpha_closure_completed_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 first-principles direct-alpha closure completed now",
            sign_base.truth(gate_a),
            "The wording sync must preserve the already-completed frozen-action direct-alpha closure rather than weaken it.",
        ),
        sign_base.row(
            "gate_b_trial2_patched_tail_pure_continuum_closure_synced_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 patched-tail pure-continuum closure synced now",
            sign_base.truth(gate_b),
            "The refreshed v2 wording now records that one patched weighted-integral continuum layer is actually closed, not merely deferred.",
        ),
        sign_base.row(
            "gate_c_trial2_operator_level_refinement_deferred_v3_and_no_unconditional_branch_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 operator-level refinement deferred to v3 and no unconditional branch now",
            sign_base.truth(gate_c),
            "After the wording sync, the only remaining open item is the full operator-level continuum theorem, so the state returns to conditional reopen only.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "alpha_common_value": float(prior_summary["alpha_common_value"]),
        "alpha_common_rel_error_vs_target": float(
            prior_summary["alpha_common_rel_error_vs_target"]
        ),
        "tail_match_x": float(prior_summary["tail_match_x"]),
        "x_cutoff": float(prior_summary["x_cutoff"]),
        "analytic_remainder_bound_n2": float(
            prior_summary["analytic_remainder_bound_n2"]
        ),
        "analytic_remainder_over_total_abs_min_n2": float(
            prior_summary["analytic_remainder_over_total_abs_min_n2"]
        ),
        "exact_trial2_first_principles_direct_alpha_closure_completed_now": bool(
            gate_a
        ),
        "exact_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_completed_now": bool(
            gate_b
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": False,
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now": bool(
            gate_c
        ),
        "trial2_patched_tail_pure_continuum_closure_refresh_lane_completed_now": bool(
            gate_b
        ),
        "no_unconditional_next_official_branch_now": bool(gate_c),
        "selected_next_generation_route": None,
        "recommended_next_route_or_none": None,
    }

    payload = {
        "step_tag": STEP_TAG,
        "step_name": STEP_NAME,
        "summary": summary,
        "rows": rows,
        "formulae": build_formulae(),
        "notes": {
            "gate_meaning": (
                "v2 wording now distinguishes the completed direct-alpha closure, the newly synced patched-tail pure-continuum closure, and the still-open operator-level continuum refinement."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
