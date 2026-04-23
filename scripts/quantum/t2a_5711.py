#!/usr/bin/env python3
"""Generate 8.7.56.5711-.5714 admissible-tail patch audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_admissible_tail_patch_followup_backend import (
    build_trial2_beta_sensitivity_admissible_tail_patch_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5707-5710",
        "updated_pack_trial2_beta_sensitivity_final_closure_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "95_trial2_numeric_alpha_vector_qball_admissible_tail_patch_followup_audit.md"
)

STEP_TAG = "8.7.56.5711-5714"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "admissible positive-decay tail patch followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_admissible_tail_patch_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_principles_direct_alpha_closure_completed_"
    "pure_analytic_refinement_deferred_v3_conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_pure_analytic_refinement_reopened_raw_tail_artifact_"
    "audited_positive_decay_patch_gate_next"
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
    """Return whether the admissible-tail note carries the expected claims."""
    patterns = (
        "raw extended tail",
        "positive-decay tail",
        "tail(22)=0",
        "artifact",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the admissible-tail audit."""
    return {
        "pivot_rule": "find_amp(beta) is fixed by tail(22)=0 in the pivot shooting solve",
        "tail_candidate": (
            "y_tail^(cand)(x) = y(x_match) * (x_match/x) * exp(-kappa * (x-x_match))"
        ),
        "tail_kappa": "kappa = sqrt(1 - beta_common_root^2)",
    }


# 関数: `.5711-.5714` を実行する。
def main() -> None:
    """Execute the admissible positive-decay tail patch audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_admissible_tail_patch_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)

    rows = [
        sign_base.row(
            "updated_pack_trial2_admissible_tail_patch_followup_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 admissible tail patch followup route selected now",
            sign_base.truth(route_selected),
            "This theorem-hardening route starts only after first-principles direct-alpha closure is complete and pure analytic refinement is the only remaining open item.",
        ),
        sign_base.row(
            "exact_trial2_admissible_tail_patch_followup_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 admissible tail patch followup note available now",
            sign_base.truth(note_available),
            "The note must explicitly separate the raw extended tail artifact from the admissible positive-decay continuation candidate.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now",
            "pass"
            if pack[
                "exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now"
            ]
            else "reject",
            "exact Trial-2 beta-sensitivity raw extended tail artifact detected now",
            sign_base.truth(
                pack[
                    "exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now"
                ]
            ),
            "Pass means the pivot shooting solve is zeroed at radius 22 while the same amplitude extended farther overshoots through zero and turns negative, so the raw post-22 tail is not admissible for theorem use.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_pivot_positive_up_to_22_now",
            "pass" if pack["pivot_positive_up_to_22_now"] else "reject",
            "exact Trial-2 beta-sensitivity pivot positive up to 22 now",
            sign_base.truth(pack["pivot_positive_up_to_22_now"]),
            "The pivot branch remains a positive localized profile up to the truncation radius where the shooting rule is applied.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now",
            "pass"
            if pack[
                "exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now"
            ]
            else "reject",
            "exact Trial-2 beta-sensitivity admissible positive-decay tail patch formula available now",
            sign_base.truth(
                pack[
                    "exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now"
                ]
            ),
            "Pass means one explicit positive-decay tail candidate exists on the linearized side beyond the potential-zero crossing and can be promoted as the next theorem-hardening front runner.",
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
            "This audit is allowed to keep the final pure analytic refinement open; it only has to isolate the admissible tail blocker honestly.",
        ),
        sign_base.row(
            "updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now",
            "pass"
            if pack[
                "updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now"
            ]
            else "reject",
            "updated-pack Trial-2 admissible positive-decay tail patch followup required now",
            sign_base.truth(
                pack[
                    "updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now"
                ]
            ),
            "Once the raw extended tail is fixed as an artifact and a positive-decay candidate exists, the next honest blocker is the patched-tail theorem itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "central_amplitude_common": float(pack["central_amplitude_common"]),
        "pivot_tail_abs_at_22": float(pack["pivot_tail_abs_at_22"]),
        "extended_profile_zero_crossing_x": float(
            pack["extended_profile_zero_crossing_x"]
        ),
        "u_beta_zero_crossing_x": float(pack["u_beta_zero_crossing_x"]),
        "potential_zero_crossing_x": float(pack["potential_zero_crossing_x"]),
        "tail_derivative_mismatch_rel": float(pack["tail_derivative_mismatch_rel"]),
        "tail_candidate_value_at_22_0": float(pack["tail_candidate_value_at_22_0"]),
        "tail_candidate_value_at_25_0": float(pack["tail_candidate_value_at_25_0"]),
        "tail_candidate_value_at_30_0": float(pack["tail_candidate_value_at_30_0"]),
        "exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now": bool(
            pack["exact_trial2_beta_sensitivity_raw_extended_tail_artifact_detected_now"]
        ),
        "exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now": bool(
            pack[
                "exact_trial2_beta_sensitivity_admissible_positive_decay_tail_patch_formula_available_now"
            ]
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": bool(
            pack[
                "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"
            ]
        ),
        "updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now": bool(
            pack[
                "updated_pack_trial2_admissible_positive_decay_tail_patch_followup_required_now"
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
                "The current pure-analytic refinement gap is no longer generic continuum support. "
                "It is the admissible treatment of the post-22 tail."
            ),
        },
    }
    written = write_artifact("declaration_gate", payload)
    print(json.dumps({"ok": True, "written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
