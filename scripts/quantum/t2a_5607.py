#!/usr/bin/env python3
"""Generate 8.7.56.5607-.5610 Trial-2 energy-partition variant audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_energy_partition_variant_backend import (
    build_trial2_energy_partition_variant_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5603-5606",
        "updated_pack_trial2_entropy_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "81_trial2_numeric_alpha_vector_qball_energy_partition_variant_audit.md"
)

STEP_TAG = "8.7.56.5607-5610"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "energy-partition variant audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_energy_partition_variant_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_entropy_negative_closeout_completed_all_promoted_direct_alpha_routes_"
    "exhausted_conditional_hold_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_energy_partition_variant_screen_interaction_total_over_harmonic_sq_"
    "front_runner_followup_gate_next"
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
    """Return whether the variant audit note carries the expected claims."""
    patterns = (
        "Variant 8",
        "interaction_total_over_harmonic_sq",
        "Variant 9",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the energy-partition variant audit."""
    return {
        "baseline": "R_base = E_int / E_harm",
        "variant_8": "R8 = (E_int / E_harm) * (E_total / E_harm)",
        "variant_9": "R9 = E_int * E_total / E_harm^2",
        "verdict": (
            "The blind screen reopens the current pack only if one variant "
            "improves the retained baseline without introducing a fit "
            "parameter."
        ),
    }


# 関数: `.5607-.5610` を実行する。

def main() -> None:
    """Execute the Trial-2 energy-partition variant audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_energy_partition_variant_pack(
        retained_beta=float(RETAINED_BETA),
        nearest_beta=float(NEAREST_BETA),
    )

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    front_runner = dict(pack["front_runner"])
    second_runner = dict(pack["second_runner"])
    baseline_row = dict(pack["baseline_row"])
    variant_8_row = dict(pack["variant_8_row"])
    variant_9_row = dict(pack["variant_9_row"])

    front_runner_selected_now = (
        str(front_runner["candidate_name"]) == "variant_8_interaction_total_over_harmonic_sq"
    )
    alias_consistent_now = bool(
        abs(float(variant_8_row["retained_value"]) - float(variant_9_row["retained_value"])) <= 1.0e-15
        and abs(float(variant_8_row["near_value"]) - float(variant_9_row["near_value"])) <= 1.0e-15
    )
    front_runner_improves_baseline_now = bool(pack["front_runner_improves_baseline_now"])
    front_runner_exact_route_available_now = bool(pack["front_runner_exact_route_available_now"])
    front_runner_exact_relation_primary_next_now = bool(
        pack["front_runner_exact_relation_primary_next_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_energy_partition_variant_reopen_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 energy-partition variant reopen selected now",
            sign_base.truth(route_selected),
            "The current pack may reopen honestly only if the new variant screen starts from the synchronized conditional-hold state.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_variant_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 energy-partition variant audit note available now",
            sign_base.truth(note_available),
            "The audit note records the blind variant list, the Variant 8 / Variant 9 equivalence, and the promoted next blocker.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_variant_front_runner_selected_now",
            "pass" if front_runner_selected_now else "reject",
            "exact Trial-2 energy-partition variant front runner selected now",
            sign_base.truth(front_runner_selected_now),
            "The blind screen should promote one unique best retained candidate rather than hand-picking a tuned expression.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_variant_8_9_alias_consistent_now",
            "pass" if alias_consistent_now else "reject",
            "exact Trial-2 energy-partition Variant 8 / Variant 9 alias consistent now",
            sign_base.truth(alias_consistent_now),
            "Variant 9 is only a symmetric rewrite of Variant 8, so both retained and near values must agree exactly.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_variant_front_runner_improves_baseline_now",
            "pass" if front_runner_improves_baseline_now else "reject",
            "exact Trial-2 energy-partition variant front runner improves baseline now",
            sign_base.truth(front_runner_improves_baseline_now),
            "A genuine reopen requires the new front runner to beat the previously closed baseline E_int / E_harm on the retained row.",
        ),
        sign_base.row(
            "exact_trial2_energy_partition_variant_exact_route_available_now",
            "pass" if front_runner_exact_route_available_now else "reject",
            "exact Trial-2 energy-partition variant exact route available now",
            sign_base.truth(front_runner_exact_route_available_now),
            "Pass would mean the promoted variant already gives one exact target-free alpha law without a new exact-relation audit.",
        ),
        sign_base.row(
            "updated_pack_trial2_energy_partition_variant_front_runner_exact_relation_primary_next_now",
            "pass" if front_runner_exact_relation_primary_next_now else "reject",
            "updated-pack Trial-2 energy-partition variant front runner exact relation primary next now",
            sign_base.truth(front_runner_exact_relation_primary_next_now),
            "Because the front runner improves the baseline but is not yet exact, the next honest blocker is its exact-relation audit.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_beta1": float(pack["retained_beta1"]),
        "nearest_alpha_beta_root_to_retained": float(pack["nearest_alpha_beta_root_to_retained"]),
        "retained_energy_kinetic": float(pack["retained_energy_row"]["energy_kinetic"]),
        "retained_energy_mass": float(pack["retained_energy_row"]["energy_mass"]),
        "retained_energy_gradient": float(pack["retained_energy_row"]["energy_gradient"]),
        "retained_energy_cubic": float(pack["retained_energy_row"]["energy_cubic"]),
        "retained_energy_quartic": float(pack["retained_energy_row"]["energy_quartic"]),
        "retained_energy_interaction": float(pack["retained_energy_row"]["energy_interaction"]),
        "retained_energy_harmonic": float(pack["retained_energy_row"]["energy_harmonic"]),
        "retained_energy_total": float(pack["retained_energy_row"]["energy_total"]),
        "energy_partition_variant_front_runner_name": str(front_runner["candidate_name"]),
        "energy_partition_variant_front_runner_formula": str(front_runner["formula"]),
        "energy_partition_variant_front_runner_retained_value": float(front_runner["retained_value"]),
        "energy_partition_variant_front_runner_retained_rel_error_vs_target": float(
            front_runner["retained_rel_error_vs_target"]
        ),
        "energy_partition_variant_front_runner_near_value": float(front_runner["near_value"]),
        "energy_partition_variant_front_runner_near_rel_error_vs_target": float(
            front_runner["near_rel_error_vs_target"]
        ),
        "energy_partition_variant_front_runner_near_rel_shift_vs_retained": float(
            front_runner["near_rel_shift_vs_retained"]
        ),
        "energy_partition_variant_second_runner_name": str(second_runner["candidate_name"]),
        "energy_partition_variant_second_runner_retained_abs_rel_error_vs_target": float(
            second_runner["retained_abs_rel_error_vs_target"]
        ),
        "energy_partition_variant_front_runner_margin_vs_second": float(
            float(second_runner["retained_abs_rel_error_vs_target"])
            - float(front_runner["retained_abs_rel_error_vs_target"])
        ),
        "baseline_interaction_over_harmonic_retained_value": float(
            baseline_row["retained_value"]
        ),
        "baseline_interaction_over_harmonic_retained_rel_error_vs_target": float(
            baseline_row["retained_rel_error_vs_target"]
        ),
        "variant_8_variant_9_alias_consistent_now": alias_consistent_now,
        "exact_trial2_energy_partition_variant_front_runner_selected_now": (
            front_runner_selected_now
        ),
        "exact_trial2_energy_partition_variant_front_runner_improves_baseline_now": (
            front_runner_improves_baseline_now
        ),
        "exact_trial2_energy_partition_variant_exact_route_available_now": (
            front_runner_exact_route_available_now
        ),
        "updated_pack_trial2_energy_partition_variant_front_runner_exact_relation_primary_next_now": (
            front_runner_exact_relation_primary_next_now
        ),
        "selected_primary_completion_lane": (
            "trial2_interaction_total_over_harmonic_sq_exact_relation"
        ),
        "selected_secondary_completion_lane": "conditional_hold_only",
        "selected_reserve_completion_lane": "conditional_hold_only",
        "selected_next_generation_route": (
            "trial2_interaction_total_over_harmonic_sq_exact_relation"
        ),
        "recommended_next_route_or_none": "8.7.56.5611",
        "selected_followup_route": "trial2_interaction_total_over_harmonic_sq_exact_relation",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5609",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT / "scripts" / "quantum" / "trial2_energy_partition_variant_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5611",
                "followup_route": "trial2_interaction_total_over_harmonic_sq_exact_relation",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_energy_partition_variant_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 energy-partition variant audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
