#!/usr/bin/env python3
"""Generate 8.7.56.5615-.5618 Trial-2 exact-relation audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_interaction_total_over_harmonic_sq_exact_relation_backend import (
    build_trial2_interaction_total_over_harmonic_sq_exact_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5611-5614",
        "updated_pack_trial2_energy_partition_variant_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "82_trial2_numeric_alpha_vector_qball_interaction_total_over_harmonic_sq_exact_relation_audit.md"
)

STEP_TAG = "8.7.56.5615-5618"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "interaction_total_over_harmonic_sq exact relation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_interaction_total_over_harmonic_sq_exact_relation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_energy_partition_variant_audited_interaction_total_over_harmonic_sq_"
    "front_runner_exact_relation_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_total_over_harmonic_sq_exact_relation_weighted_eom_"
    "one_third_factor_local_beta_root_followup_gate_next"
)
RETAINED_BETA = 0.9982557379261291
NEAREST_BETA = 0.9982996989044647


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


# 関数: audit note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the exact-relation note carries the expected claims."""
    patterns = (
        "1/3",
        "weighted-EOM identity",
        "local beta root",
        "R_8",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the exact-relation audit."""
    return {
        "weighted_eom": "B - I_g - epsilon_beta I2 + 3 I3 + I4 = 0",
        "exact_relation": (
            "R8 = ([4(I_g + epsilon_beta I2 - B) - I4] * "
            "[2(5+beta^2)I2 + 10 I_g - I4 - 4B]) / "
            "[36(1+beta^2)^2 I2^2]"
        ),
        "leading_relation": (
            "R8_LO = (4(I_g + epsilon_beta I2 - B) * "
            "[2(5+beta^2)I2 + 10 I_g - 4B]) / "
            "[36(1+beta^2)^2 I2^2]"
        ),
    }


# 関数: `.5615-.5618` を実行する。

def main() -> None:
    """Execute the Trial-2 interaction_total_over_harmonic_sq exact audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_interaction_total_over_harmonic_sq_exact_pack(
        retained_beta=float(RETAINED_BETA),
        nearest_beta=float(NEAREST_BETA),
    )

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    retained = dict(pack["retained_row"])
    near_row = dict(pack["nearest_row"])

    exact_weighted_eom_identity_available_now = bool(
        pack["exact_weighted_eom_identity_available_now"]
    )
    exact_relation_available_now = bool(pack["exact_relation_available_now"])
    one_third_factor_explicit_now = bool(pack["one_third_factor_explicit_now"])
    quartic_negligible_now = bool(pack["quartic_negligible_now"])
    leading_relation_subpercent_now = bool(pack["leading_relation_subpercent_now"])
    local_beta_root_available_now = bool(pack["local_beta_root_available_now"])
    beta_root_consistent_with_prior_alpha_beta_now = bool(
        pack["beta_root_consistent_with_prior_alpha_beta_now"]
    )
    exact_target_free_closeout_available_now = bool(
        pack["exact_target_free_closeout_available_now"]
    )
    beta_root_followup_required_now = bool(pack["beta_root_followup_required_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_interaction_total_over_harmonic_sq_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 interaction_total_over_harmonic_sq route selected now",
            sign_base.truth(route_selected),
            "The exact-relation audit starts only from the synchronized variant gate that promoted interaction_total_over_harmonic_sq as the current front runner.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq audit note available now",
            sign_base.truth(note_available),
            "The note must record the weighted-EOM identity, the explicit 1/3 factor, and the local beta-root implication.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_weighted_eom_identity_available_now",
            "pass" if exact_weighted_eom_identity_available_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq weighted-EOM identity available now",
            sign_base.truth(exact_weighted_eom_identity_available_now),
            "The retained and near rows must satisfy the finite-radius weighted-EOM identity before the cubic elimination can be trusted.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_exact_relation_available_now",
            "pass" if exact_relation_available_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq exact relation available now",
            sign_base.truth(exact_relation_available_now),
            "The eliminated expression must reconstruct the screened ratio on both retained and near rows to numerical precision.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_one_third_factor_explicit_now",
            "pass" if one_third_factor_explicit_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq one-third factor explicit now",
            sign_base.truth(one_third_factor_explicit_now),
            "The Mexican-hat cubic coefficient must appear as an explicit 1/3 after eliminating I3 from the weighted-EOM identity.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_quartic_negligible_now",
            "pass" if quartic_negligible_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq quartic negligible now",
            sign_base.truth(quartic_negligible_now),
            "The cubic-dominant interpretation is only honest if E_cubic continues to saturate the interaction energy on both rows.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_leading_relation_subpercent_now",
            "pass" if leading_relation_subpercent_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq leading relation subpercent now",
            sign_base.truth(leading_relation_subpercent_now),
            "The cubic-dominant leading relation should remain sub-percent as a meaningful simplification even if it is not exact.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_local_beta_root_available_now",
            "pass" if local_beta_root_available_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq local beta root available now",
            sign_base.truth(local_beta_root_available_now),
            "An exact-relation followup is only alive if the beta-family itself exhibits one local root near the retained row.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_beta_root_consistent_with_prior_alpha_beta_now",
            "pass" if beta_root_consistent_with_prior_alpha_beta_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq beta root consistent with prior alpha(beta) now",
            sign_base.truth(beta_root_consistent_with_prior_alpha_beta_now),
            "The new exact-relation root should agree with the previously audited alpha(beta) microshift root to tight relative tolerance.",
        ),
        sign_base.row(
            "exact_trial2_interaction_total_over_harmonic_sq_target_free_closeout_available_now",
            "pass" if exact_target_free_closeout_available_now else "reject",
            "exact Trial-2 interaction_total_over_harmonic_sq target-free closeout available now",
            sign_base.truth(exact_target_free_closeout_available_now),
            "Pass would mean the exact relation itself already selects alpha without any comparator-driven beta-root followup.",
        ),
        sign_base.row(
            "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_next_now",
            "pass" if beta_root_followup_required_now else "reject",
            "updated-pack Trial-2 interaction_total_over_harmonic_sq beta-root followup primary next now",
            sign_base.truth(beta_root_followup_required_now),
            "Because the exact relation is real but target-free closeout is still unavailable, the next honest blocker is the beta-root followup.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_beta1": float(pack["retained_beta1"]),
        "nearest_alpha_beta_root_to_retained": float(pack["nearest_alpha_beta_root_to_retained"]),
        "prior_alpha_beta_root": float(pack["prior_alpha_beta_root"]),
        "retained_i2": float(retained["i2"]),
        "retained_ig": float(retained["ig"]),
        "retained_i3": float(retained["i3"]),
        "retained_i4": float(retained["i4"]),
        "retained_boundary_weighted_eom": float(retained["boundary_weighted_eom"]),
        "retained_weighted_eom_residual": float(retained["weighted_eom_residual"]),
        "retained_exact_relation_value": float(retained["exact_relation_from_integrals"]),
        "retained_exact_relation_rel_error_vs_target": float(
            retained["exact_relation_rel_error_vs_target"]
        ),
        "retained_exact_relation_from_weighted_eom": float(
            retained["exact_relation_from_weighted_eom"]
        ),
        "retained_exact_relation_weighted_eom_residual": float(
            retained["exact_relation_weighted_eom_residual"]
        ),
        "retained_leading_relation_cubic_dominant": float(
            retained["leading_relation_cubic_dominant"]
        ),
        "retained_leading_relation_rel_error_vs_target": float(
            retained["leading_relation_rel_error_vs_target"]
        ),
        "retained_cubic_share_of_interaction": float(retained["cubic_share_of_interaction"]),
        "near_exact_relation_value": float(near_row["exact_relation_from_integrals"]),
        "near_exact_relation_rel_error_vs_target": float(
            near_row["exact_relation_rel_error_vs_target"]
        ),
        "near_leading_relation_cubic_dominant": float(
            near_row["leading_relation_cubic_dominant"]
        ),
        "near_leading_relation_rel_error_vs_target": float(
            near_row["leading_relation_rel_error_vs_target"]
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_weighted_eom_identity_available_now": (
            exact_weighted_eom_identity_available_now
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_exact_relation_available_now": (
            exact_relation_available_now
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_one_third_factor_explicit_now": (
            one_third_factor_explicit_now
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_quartic_negligible_now": (
            quartic_negligible_now
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_leading_relation_subpercent_now": (
            leading_relation_subpercent_now
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_leading_relation_point_one_percent_now": (
            bool(pack["leading_relation_point_one_percent_now"])
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_local_beta_root_available_now": (
            local_beta_root_available_now
        ),
        "interaction_total_over_harmonic_sq_beta_root": float(pack["beta_root"]),
        "interaction_total_over_harmonic_sq_beta_root_rel_shift_vs_retained": float(
            pack["beta_root_rel_shift_vs_retained"]
        ),
        "interaction_total_over_harmonic_sq_beta_root_rel_shift_vs_prior_alpha_beta": float(
            pack["beta_root_rel_shift_vs_prior_alpha_beta"]
        ),
        "interaction_total_over_harmonic_sq_beta_root_exact_relation_rel_error_vs_target": float(
            pack["beta_root_exact_relation_rel_error_vs_target"]
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_beta_root_consistent_with_prior_alpha_beta_now": (
            beta_root_consistent_with_prior_alpha_beta_now
        ),
        "exact_trial2_interaction_total_over_harmonic_sq_target_free_closeout_available_now": (
            exact_target_free_closeout_available_now
        ),
        "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_followup_primary_next_now": (
            beta_root_followup_required_now
        ),
        "selected_primary_completion_lane": (
            "trial2_interaction_total_over_harmonic_sq_beta_root_followup"
        ),
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": (
            "trial2_interaction_total_over_harmonic_sq_beta_root_followup"
        ),
        "recommended_next_route_or_none": "8.7.56.5619",
        "selected_followup_route": "trial2_interaction_total_over_harmonic_sq_beta_root_followup",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5617",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT
                    / "scripts"
                    / "quantum"
                    / "trial2_interaction_total_over_harmonic_sq_exact_relation_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5619",
                "followup_route": "trial2_interaction_total_over_harmonic_sq_beta_root_followup",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_interaction_total_over_harmonic_sq_exact_relation_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 interaction_total_over_harmonic_sq exact audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
