#!/usr/bin/env python3
"""Generate 8.7.56.5715-.5718 admissible-tail patch gate artifacts."""

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
        "8.7.56.5711-5714",
        "updated_pack_trial2_beta_sensitivity_admissible_tail_patch_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5715-5718"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "admissible positive-decay tail patch gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_admissible_tail_patch_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_pure_analytic_refinement_reopened_raw_tail_artifact_"
    "audited_positive_decay_patch_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_pure_analytic_refinement_reopened_raw_tail_artifact_detected_"
    "positive_decay_tail_patch_primary_conditional_reopen_secondary_next"
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
    """Return formulas used by the admissible-tail gate."""
    return {
        "gate_a": "Gate A = raw extended tail artifact is detected now",
        "gate_b": "Gate B = admissible positive-decay tail patch formula is available now",
        "gate_c": "Gate C = pure analytic continuum refinement is reopened on the patched-tail route",
    }


# 関数: `.5715-.5718` を実行する。
def main() -> None:
    """Execute the admissible positive-decay tail patch gate / refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now"
        ]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now"
        ]
    )
    gate_c = bool(
        gate_b
        and not prior_summary[
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"
        ]
    )

    rows = [
        sign_base.row(
            "gate_a_trial2_raw_extended_tail_artifact_detected_now",
            "pass" if gate_a else "reject",
            "gate A Trial-2 raw extended tail artifact detected now",
            sign_base.truth(gate_a),
            "The route refresh starts only once the raw post-22 extension is fixed as inadmissible for theorem use.",
        ),
        sign_base.row(
            "gate_b_trial2_admissible_positive_decay_tail_patch_formula_available_now",
            "pass" if gate_b else "reject",
            "gate B Trial-2 admissible positive-decay tail patch formula available now",
            sign_base.truth(gate_b),
            "The reopened refinement route needs one explicit positive-decay tail candidate before any patched-tail theorem can proceed.",
        ),
        sign_base.row(
            "gate_c_trial2_pure_analytic_refinement_reopened_now",
            "pass" if gate_c else "reject",
            "gate C Trial-2 pure analytic refinement reopened now",
            sign_base.truth(gate_c),
            "Once the artifact and patch candidate are both fixed, the next honest blocker becomes the patched-tail theorem rather than unconditional hold.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(prior_summary["beta_common_root"]),
        "exact_trial2_first_principles_direct_alpha_closure_completed_now": True,
        "exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now": bool(
            gate_a
        ),
        "exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now": bool(
            gate_b
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": False,
        "updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now": bool(
            gate_c
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_admissible_positive_decay_tail_patch_followup"
            if gate_c
            else None
        ),
        "recommended_next_route_or_none": (
            "8.7.56.5719-5722"
            if gate_c
            else None
        ),
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
                "Theorem hardening reopens from conditional hold specifically on the admissible positive-decay tail patch route."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
