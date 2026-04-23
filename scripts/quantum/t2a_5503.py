#!/usr/bin/env python3
"""Generate 8.7.56.5503-.5506 Trial-2 spectral distinguished-scale audit artifacts."""

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
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5499-5502",
        "updated_pack_trial2_blind_overlap_theorem_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
MATCHING_GATE = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_declaration_gate_metrics.json"
)
SUPPORT_GATE = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review_declaration_gate_metrics.json"
)
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "67_trial2_numeric_alpha_vector_qball_spectral_distinguished_scale_audit.md"
)

STEP_TAG = "8.7.56.5503-5506"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "spectral distinguished-scale audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_spectral_distinguished_scale_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_blind_overlap_theorem_target_free_negative_closeout_completed_"
    "spectral_primary_residue_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_spectral_distinguished_scale_target_free_negative_closeout_"
    "completed_residue_primary_source_materialization_reserve_gate_next"
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


# 関数: spectral audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the spectral distinguished-scale audit."""
    return {
        "support_band": "q lies in the finite internal support band justified by the spherical-kernel phase structure",
        "selector_requirement": "S_spec[f0] = q_exact without alpha_target and without externally chosen support-scale rule",
        "no_go": "finite support band justified but one unique distinguished spectral scale selector absent",
    }


# 関数: note が expected spectral audit claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the note carries the expected spectral-audit claims."""
    patterns = (
        "spectral distinguished-scale route",
        "finite support band",
        "one target-free distinguished spectral scale selector is unavailable",
        "negative closeout",
        "effective coupling / residue route",
    )
    return all(pattern in text for pattern in patterns)


# 関数: `.5503-.5506` を実行する。

def main() -> None:
    """Execute the Trial-2 spectral distinguished-scale audit."""
    for path in (PRIOR_GATE, MATCHING_GATE, SUPPORT_GATE, AUDIT_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    matching_payload = sign_base.read_json(MATCHING_GATE)
    support_payload = sign_base.read_json(SUPPORT_GATE)
    matching_summary = matching_payload["summary"]
    support_summary = support_payload["summary"]
    support_evidence = support_payload["evidence"]
    note_text = sign_base.read_text(AUDIT_NOTE)

    note_available = note_contains_audit(note_text)
    spectral_support_band_available_now = bool(
        matching_summary["finite_internal_scale_theory_side_justified"]
        and support_summary["finite_internal_support_band_justified"]
    )
    spectral_target_free_selector_available_now = bool(
        support_summary["unique_effective_support_scale_available"]
    )
    spectral_current_public_nonuniqueness_surface_available_now = bool(
        support_summary["current_public_nonuniqueness_surface_available"]
    )
    spectral_target_free_theorem_available_now = bool(
        spectral_support_band_available_now
        and spectral_target_free_selector_available_now
    )
    spectral_distinguished_scale_lane_negative_closeout_available_now = bool(
        spectral_support_band_available_now
        and spectral_current_public_nonuniqueness_surface_available_now
        and not spectral_target_free_theorem_available_now
    )
    updated_pack_trial2_effective_coupling_residue_primary_followup_required_now = (
        bool(spectral_distinguished_scale_lane_negative_closeout_available_now)
    )

    best_candidate_name = support_evidence["effective_support_scale_audit_summary"][
        "best_candidate_name"
    ]
    best_candidate_error = float(
        support_evidence["effective_support_scale_audit_summary"]["best_candidate_error"]
    )
    second_candidate_name = support_evidence["effective_support_scale_audit_summary"][
        "second_candidate_name"
    ]
    second_candidate_error = float(
        support_evidence["effective_support_scale_audit_summary"][
            "second_candidate_error"
        ]
    )

    rows = [
        sign_base.row(
            "exact_trial2_spectral_distinguished_scale_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 spectral distinguished-scale audit note available now",
            sign_base.truth(note_available),
            "The dedicated spectral-route audit note exists and states the current support-band/no-go split.",
        ),
        sign_base.row(
            "exact_trial2_spectral_distinguished_scale_support_band_available_now",
            "pass" if spectral_support_band_available_now else "reject",
            "exact Trial-2 spectral distinguished-scale support band available now",
            sign_base.truth(spectral_support_band_available_now),
            "Current canon still supports a finite internal support band for the retained matching scale.",
        ),
        sign_base.row(
            "exact_trial2_spectral_distinguished_scale_target_free_selector_available_now",
            "pass" if spectral_target_free_selector_available_now else "reject",
            "exact Trial-2 spectral distinguished-scale target-free selector available now",
            sign_base.truth(spectral_target_free_selector_available_now),
            "A strict theorem would require one unique distinguished spectral-scale selector, but the old support-scale review keeps this unavailable.",
        ),
        sign_base.row(
            "exact_trial2_spectral_distinguished_scale_public_nonuniqueness_surface_available_now",
            "pass"
            if spectral_current_public_nonuniqueness_surface_available_now
            else "reject",
            "exact Trial-2 spectral distinguished-scale public nonuniqueness surface available now",
            sign_base.truth(spectral_current_public_nonuniqueness_surface_available_now),
            "Current canon still exposes multiple comparable internal-scale candidates rather than one distinguished selector.",
        ),
        sign_base.row(
            "exact_trial2_spectral_distinguished_scale_target_free_theorem_available_now",
            "pass" if spectral_target_free_theorem_available_now else "reject",
            "exact Trial-2 spectral distinguished-scale target-free theorem available now",
            sign_base.truth(spectral_target_free_theorem_available_now),
            "Reject means the route stops at a justified support band and does not yet choose one target-free distinguished scale.",
        ),
        sign_base.row(
            "exact_trial2_spectral_distinguished_scale_lane_negative_closeout_available_now",
            "pass"
            if spectral_distinguished_scale_lane_negative_closeout_available_now
            else "reject",
            "exact Trial-2 spectral distinguished-scale lane negative closeout available now",
            sign_base.truth(
                spectral_distinguished_scale_lane_negative_closeout_available_now
            ),
            "The spectral route closes honestly as a strict-theorem no-go under the current canon limit.",
        ),
        sign_base.row(
            "updated_pack_trial2_effective_coupling_residue_primary_followup_required_now",
            "pass"
            if updated_pack_trial2_effective_coupling_residue_primary_followup_required_now
            else "reject",
            "updated-pack Trial-2 effective coupling / residue primary followup required now",
            sign_base.truth(
                updated_pack_trial2_effective_coupling_residue_primary_followup_required_now
            ),
            "Once the spectral route closes negatively, the honest next primary route is effective coupling / residue.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "best_spectral_candidate_name": best_candidate_name,
        "best_spectral_candidate_relative_error": best_candidate_error,
        "second_spectral_candidate_name": second_candidate_name,
        "second_spectral_candidate_relative_error": second_candidate_error,
        "exact_trial2_spectral_distinguished_scale_audit_note_available_now": (
            note_available
        ),
        "exact_trial2_spectral_distinguished_scale_support_band_available_now": (
            spectral_support_band_available_now
        ),
        "exact_trial2_spectral_distinguished_scale_target_free_selector_available_now": (
            spectral_target_free_selector_available_now
        ),
        "exact_trial2_spectral_distinguished_scale_public_nonuniqueness_surface_available_now": (
            spectral_current_public_nonuniqueness_surface_available_now
        ),
        "exact_trial2_spectral_distinguished_scale_target_free_theorem_available_now": (
            spectral_target_free_theorem_available_now
        ),
        "exact_trial2_spectral_distinguished_scale_lane_negative_closeout_available_now": (
            spectral_distinguished_scale_lane_negative_closeout_available_now
        ),
        "updated_pack_trial2_effective_coupling_residue_primary_followup_required_now": (
            updated_pack_trial2_effective_coupling_residue_primary_followup_required_now
        ),
        "selected_primary_completion_lane": "effective_coupling_residue",
        "selected_secondary_completion_lane": "selected_extension_source_materialization",
        "selected_reserve_completion_lane": "none_current_pack",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "effective_coupling_residue_primary"
        ),
        "recommended_next_route_or_none": "8.7.56.5507",
        "selected_followup_route": "effective_coupling_residue",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5505",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "matching_gate": sign_base.display_path(MATCHING_GATE),
                "support_gate": sign_base.display_path(SUPPORT_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5507",
                "followup_route": "effective_coupling_residue",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_spectral_distinguished_scale_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 spectral distinguished-scale audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から spectral distinguished-scale audit を実行する。

if __name__ == "__main__":
    main()
