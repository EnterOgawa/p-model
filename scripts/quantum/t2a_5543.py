#!/usr/bin/env python3
"""Generate 8.7.56.5543-.5546 Trial-2 full spectral / Jost audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_full_spectral_jost_route_backend import (
    build_trial2_full_spectral_jost_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5539-5542",
        "updated_pack_trial2_bohr_radius_matching_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "72_trial2_numeric_alpha_vector_qball_full_spectral_jost_audit.md"
)

STEP_TAG = "8.7.56.5543-5546"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball full spectral / Jost audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_full_spectral_jost_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_bohr_radius_matching_negative_closeout_completed_full_spectral_jost_"
    "primary_scattering_thomson_secondary_ward_current_algebra_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_full_spectral_jost_target_free_selector_missing_scattering_thomson_"
    "primary_gate_next"
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


# 関数: audit note が expected Jost claims を含むか確認する。

def note_contains_jost_audit(text: str) -> bool:
    """Return whether the full spectral / Jost note carries the expected claims."""
    patterns = (
        "full spectral / Jost",
        "unique s-wave radial operator",
        "Born Jost proxy",
        "exact s-wave phase shift",
        "negative closeout",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the full spectral / Jost audit."""
    return {
        "linearized_operator": "L0 = -d^2/dx^2 + (epsilon_beta - 6 y(x) - 3 y(x)^2)",
        "shifted_potential": "U0(x) = -6 y(x) - 3 y(x)^2",
        "born_jost": "J_Born(q) = 1 + (1/(2 i k(q))) int (e^{2 i k(q) x} - 1) U0(x) dx",
        "phase_peak": "q_delta_peak = argmax_q delta_0(q)",
    }


# 関数: `.5543-.5546` を実行する。

def main() -> None:
    """Execute the Trial-2 full spectral / Jost audit."""
    for path in (PRIOR_GATE, AUDIT_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    audit_note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_full_spectral_jost_pack()

    note_available = note_contains_jost_audit(audit_note_text)
    operator_available_now = bool(pack["s_wave_operator_available_now"])
    exact_phase_surface_available_now = bool(pack["exact_phase_surface_available_now"])
    exact_jost_function_materialized_now = bool(pack["exact_jost_function_materialized_now"])
    target_free_selector_available_now = bool(pack["target_free_selector_available_now"])
    full_spectral_jost_lane_negative_closeout_available_now = bool(
        pack["negative_closeout_available_now"]
    )
    updated_pack_trial2_scattering_thomson_primary_followup_required_now = bool(
        full_spectral_jost_lane_negative_closeout_available_now
        and not target_free_selector_available_now
    )

    rows = [
        sign_base.row(
            "exact_trial2_full_spectral_jost_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 full spectral / Jost audit note available now",
            sign_base.truth(note_available),
            "The dedicated note exists and records the operator-level spectral audit rather than a support-band replay.",
        ),
        sign_base.row(
            "exact_trial2_full_spectral_jost_operator_available_now",
            "pass" if operator_available_now else "reject",
            "exact Trial-2 full spectral / Jost operator available now",
            sign_base.truth(operator_available_now),
            "The unique linearized s-wave radial operator is materialized from the retained frozen-action profile.",
        ),
        sign_base.row(
            "exact_trial2_full_spectral_jost_exact_phase_surface_available_now",
            "pass" if exact_phase_surface_available_now else "reject",
            "exact Trial-2 full spectral / Jost exact phase surface available now",
            sign_base.truth(exact_phase_surface_available_now),
            "The exact s-wave phase shift of the retained operator can be evaluated directly.",
        ),
        sign_base.row(
            "exact_trial2_full_spectral_jost_exact_jost_function_materialized_now",
            "pass" if exact_jost_function_materialized_now else "reject",
            "exact Trial-2 full spectral / Jost exact Jost function materialized now",
            sign_base.truth(exact_jost_function_materialized_now),
            "Reject means the route still relies on a Born Jost proxy rather than an exact Jost theorem object.",
        ),
        sign_base.row(
            "exact_trial2_full_spectral_jost_phase_peak_beats_q_star_now",
            "pass" if pack["phase_peak_beats_q_star_now"] else "reject",
            "exact Trial-2 full spectral / Jost phase peak beats q_star now",
            sign_base.truth(pack["phase_peak_beats_q_star_now"]),
            "The exact phase peak would need to improve on retained q_star to become a credible theorem selector.",
        ),
        sign_base.row(
            "exact_trial2_full_spectral_jost_landmark_nonuniqueness_now",
            "pass" if pack["spectral_landmark_nonuniqueness_now"] else "reject",
            "exact Trial-2 full spectral / Jost landmark nonuniqueness now",
            sign_base.truth(pack["spectral_landmark_nonuniqueness_now"]),
            "Born Jost and exact phase landmarks do not collapse to one shared distinguished scale.",
        ),
        sign_base.row(
            "exact_trial2_full_spectral_jost_target_free_selector_available_now",
            "pass" if target_free_selector_available_now else "reject",
            "exact Trial-2 full spectral / Jost target-free selector available now",
            sign_base.truth(target_free_selector_available_now),
            "Reject means the canonical operator-level landmarks still fail to select q_exact target-free.",
        ),
        sign_base.row(
            "exact_trial2_full_spectral_jost_lane_negative_closeout_available_now",
            "pass" if full_spectral_jost_lane_negative_closeout_available_now else "reject",
            "exact Trial-2 full spectral / Jost lane negative closeout available now",
            sign_base.truth(full_spectral_jost_lane_negative_closeout_available_now),
            "The route now closes honestly as operator available but theorem selector unavailable.",
        ),
        sign_base.row(
            "updated_pack_trial2_scattering_thomson_primary_followup_required_now",
            "pass" if updated_pack_trial2_scattering_thomson_primary_followup_required_now else "reject",
            "updated-pack Trial-2 scattering / Thomson primary followup required now",
            sign_base.truth(updated_pack_trial2_scattering_thomson_primary_followup_required_now),
            "Once Jost closes negatively, the honest next primary route is scattering / Thomson-limit.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "q_star_rel_error_vs_q_exact": float(pack["q_star_rel_error_vs_q_exact"]),
        "threshold_q_over_m0": float(pack["threshold_q_over_m0"]),
        "s_wave_potential_min": float(pack["s_wave_potential_min"]),
        "s_wave_negative_area": float(pack["s_wave_negative_area"]),
        "born_re_jost_zero_q_over_m0": float(pack["born_re_jost_zero_q_over_m0"]),
        "born_re_jost_zero_rel_error_vs_q_exact": float(
            pack["born_re_jost_zero_rel_error_vs_q_exact"]
        ),
        "exact_phase_peak_q_over_m0": float(pack["exact_phase_peak_q_over_m0"]),
        "exact_phase_peak_value": float(pack["exact_phase_peak_value"]),
        "exact_phase_peak_rel_error_vs_q_exact": float(
            pack["exact_phase_peak_rel_error_vs_q_exact"]
        ),
        "exact_phase_derivative_peak_q_over_m0": float(
            pack["exact_phase_derivative_peak_q_over_m0"]
        ),
        "exact_phase_derivative_peak_rel_error_vs_q_exact": float(
            pack["exact_phase_derivative_peak_rel_error_vs_q_exact"]
        ),
        "exact_trial2_full_spectral_jost_audit_note_available_now": note_available,
        "exact_trial2_full_spectral_jost_operator_available_now": operator_available_now,
        "exact_trial2_full_spectral_jost_exact_phase_surface_available_now": (
            exact_phase_surface_available_now
        ),
        "exact_trial2_full_spectral_jost_exact_jost_function_materialized_now": (
            exact_jost_function_materialized_now
        ),
        "exact_trial2_full_spectral_jost_target_free_selector_available_now": (
            target_free_selector_available_now
        ),
        "exact_trial2_full_spectral_jost_landmark_nonuniqueness_now": (
            pack["spectral_landmark_nonuniqueness_now"]
        ),
        "exact_trial2_full_spectral_jost_lane_negative_closeout_available_now": (
            full_spectral_jost_lane_negative_closeout_available_now
        ),
        "updated_pack_trial2_scattering_thomson_primary_followup_required_now": (
            updated_pack_trial2_scattering_thomson_primary_followup_required_now
        ),
        "selected_primary_completion_lane": "trial2_scattering_thomson",
        "selected_secondary_completion_lane": "trial2_ward_current_algebra",
        "selected_reserve_completion_lane": "none_current_pack",
        "selected_next_generation_route": "trial2_scattering_thomson",
        "recommended_next_route_or_none": "8.7.56.5547",
        "selected_followup_route": "trial2_scattering_thomson",
        "selected_followup_route_or_none": "8.7.56.5551",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5545",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5547",
                "followup_route": "8.7.56.5551",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_full_spectral_jost_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae(), "evidence_pack": pack},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 full spectral / Jost audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から full spectral / Jost audit を実行する。

if __name__ == "__main__":
    main()
