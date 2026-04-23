#!/usr/bin/env python3
"""Generate 8.7.56.5527-.5530 Trial-2 fresh-pattern Round-1 audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_fresh_pattern_round1_backend import (
    build_trial2_fresh_pattern_round1_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5523-5526",
        "updated_pack_trial2_new_route_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
ROUND1_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "70_trial2_numeric_alpha_vector_qball_fresh_pattern_round1_audit.md"
)

STEP_TAG = "8.7.56.5527-5530"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fresh pattern Round-1 audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_fresh_pattern_round1_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_new_route_inventory_audited_full_spectral_jost_primary_"
    "scattering_thomson_secondary_ward_current_algebra_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_fresh_pattern_round1_audited_bohr_radius_front_runner_"
    "full_spectral_jost_secondary_scattering_thomson_reserve_gate_next"
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


# 関数: Round-1 note が expected pattern claims を含むかを確認する。

def note_contains_round1_patterns(text: str) -> bool:
    """Return whether the note carries the expected Round-1 pattern inventory."""
    patterns = (
        "Bohr radius / Compton wavelength matching",
        "mean momentum transfer",
        "characteristic-point",
        "nonlinear dispersion relation",
        "front-runner",
    )
    return all(pattern in text for pattern in patterns)


# 関数: fresh-pattern audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the fresh-pattern Round-1 audit."""
    return {
        "pattern_epsilon": "alpha_epsilon = sqrt(1 - beta1^2) / 8",
        "pattern_zeta": "<q> = int q |F(q)|^2 q^2 dq / int |F(q)|^2 q^2 dq",
        "pattern_gamma": "characteristic-point screen = {F'(q_exact), F''(q_exact), nearest inflection}",
        "pattern_beta": "kappa_NL^2 = epsilon_beta + 6 <y^2>_rho",
    }


# 関数: `.5527-.5530` を実行する。

def main() -> None:
    """Execute the Trial-2 fresh-pattern Round-1 audit."""
    for path in (PRIOR_GATE, ROUND1_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    round1_pack = build_trial2_fresh_pattern_round1_pack()
    round1_note_text = sign_base.read_text(ROUND1_NOTE)

    note_available = note_contains_round1_patterns(round1_note_text)
    pattern_epsilon_available_now = bool(
        note_available and round1_pack["alpha_bohr_root_count"] == 1
    )
    pattern_epsilon_front_runner_now = bool(round1_pack["pattern_epsilon_front_runner_now"])
    pattern_zeta_negative_screen_now = bool(round1_pack["pattern_zeta_negative_screen_now"])
    pattern_gamma_negative_screen_now = bool(round1_pack["pattern_gamma_negative_screen_now"])
    pattern_beta_negative_screen_now = bool(round1_pack["pattern_beta_negative_screen_now"])
    fresh_pattern_round1_inventory_nonempty_now = bool(
        note_available
        and pattern_epsilon_available_now
        and pattern_zeta_negative_screen_now
        and pattern_gamma_negative_screen_now
        and pattern_beta_negative_screen_now
    )
    updated_pack_pattern_epsilon_followup_required_now = bool(
        fresh_pattern_round1_inventory_nonempty_now and pattern_epsilon_front_runner_now
    )

    rows = [
        sign_base.row(
            "exact_trial2_fresh_pattern_round1_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 fresh-pattern Round-1 note available now",
            sign_base.truth(note_available),
            "The dedicated expert note exists and states epsilon/zeta/gamma/beta screening.",
        ),
        sign_base.row(
            "trial2_pattern_epsilon_bohr_available_now",
            "pass" if pattern_epsilon_available_now else "reject",
            "Trial-2 pattern epsilon Bohr / Compton candidate available now",
            sign_base.truth(pattern_epsilon_available_now),
            "The epsilon candidate yields one unique retained alpha(q) crossing without alpha_target as an input.",
        ),
        sign_base.row(
            "trial2_pattern_epsilon_front_runner_now",
            "pass" if pattern_epsilon_front_runner_now else "reject",
            "Trial-2 pattern epsilon front-runner now",
            sign_base.truth(pattern_epsilon_front_runner_now),
            "Its q-candidate improves on q_star and is the only surviving low-cost fresh pattern in Round-1.",
        ),
        sign_base.row(
            "trial2_pattern_zeta_negative_screen_now",
            "pass" if pattern_zeta_negative_screen_now else "reject",
            "Trial-2 pattern zeta negative screen now",
            sign_base.truth(pattern_zeta_negative_screen_now),
            "Mean momentum transfer is window-stable but remains far from q_exact.",
        ),
        sign_base.row(
            "trial2_pattern_gamma_negative_screen_now",
            "pass" if pattern_gamma_negative_screen_now else "reject",
            "Trial-2 pattern gamma negative screen now",
            sign_base.truth(pattern_gamma_negative_screen_now),
            "q_exact is neither a stationary point nor a nearby inflection point of F(q).",
        ),
        sign_base.row(
            "trial2_pattern_beta_negative_screen_now",
            "pass" if pattern_beta_negative_screen_now else "reject",
            "Trial-2 pattern beta negative screen now",
            sign_base.truth(pattern_beta_negative_screen_now),
            "The natural global nonlinear-dispersion correction pushes q in the wrong direction.",
        ),
        sign_base.row(
            "trial2_fresh_pattern_round1_inventory_nonempty_now",
            "pass" if fresh_pattern_round1_inventory_nonempty_now else "reject",
            "Trial-2 fresh-pattern Round-1 inventory nonempty now",
            sign_base.truth(fresh_pattern_round1_inventory_nonempty_now),
            "The expert-supplied fresh patterns have now been screened into one surviving front-runner and three negative screens.",
        ),
        sign_base.row(
            "updated_pack_pattern_epsilon_followup_required_now",
            "pass" if updated_pack_pattern_epsilon_followup_required_now else "reject",
            "updated-pack pattern epsilon followup required now",
            sign_base.truth(updated_pack_pattern_epsilon_followup_required_now),
            "The next honest blocker is not generic Jost replay but why the 1/8 Bohr / Compton ratio should hold target-free.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(round1_pack["q_exact_over_m0"]),
        "q_star_over_m0": float(round1_pack["q_star_over_m0"]),
        "q_star_relative_error_vs_q_exact": float(
            round1_pack["q_star_relative_error_vs_q_exact"]
        ),
        "q_bohr_over_m0": float(round1_pack["q_bohr_over_m0"]),
        "q_bohr_relative_error_vs_q_exact": float(
            round1_pack["q_bohr_relative_error_vs_q_exact"]
        ),
        "alpha_exact_from_q_exact": float(round1_pack["alpha_exact_from_q_exact"]),
        "alpha_bohr_one_eighth": float(round1_pack["alpha_bohr_one_eighth"]),
        "alpha_bohr_relative_error_vs_exact": float(
            round1_pack["alpha_bohr_relative_error_vs_exact"]
        ),
        "best_mean_q_over_m0": float(round1_pack["best_mean_q_over_m0"]),
        "best_mean_q_relative_error_vs_q_exact": float(
            round1_pack["best_mean_q_relative_error_vs_q_exact"]
        ),
        "log_derivative_at_q_exact": float(round1_pack["log_derivative_at_q_exact"]),
        "nearest_inflection_distance_over_m0": float(
            round1_pack["nearest_inflection_distance_over_m0"]
        ),
        "q_nonlinear_dispersion_over_m0": float(
            round1_pack["q_nonlinear_dispersion_over_m0"]
        ),
        "q_nonlinear_dispersion_relative_error_vs_q_exact": float(
            round1_pack["q_nonlinear_dispersion_relative_error_vs_q_exact"]
        ),
        "exact_trial2_fresh_pattern_round1_note_available_now": note_available,
        "trial2_pattern_epsilon_bohr_available_now": pattern_epsilon_available_now,
        "trial2_pattern_epsilon_front_runner_now": pattern_epsilon_front_runner_now,
        "trial2_pattern_zeta_negative_screen_now": pattern_zeta_negative_screen_now,
        "trial2_pattern_gamma_negative_screen_now": pattern_gamma_negative_screen_now,
        "trial2_pattern_beta_negative_screen_now": pattern_beta_negative_screen_now,
        "trial2_fresh_pattern_round1_inventory_nonempty_now": (
            fresh_pattern_round1_inventory_nonempty_now
        ),
        "updated_pack_pattern_epsilon_followup_required_now": (
            updated_pack_pattern_epsilon_followup_required_now
        ),
        "selected_primary_completion_lane": "pattern_epsilon_bohr_radius_matching",
        "selected_secondary_completion_lane": "full_spectral_jost",
        "selected_reserve_completion_lane": "scattering_thomson_then_ward_current_algebra",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_pattern_epsilon_"
            "bohr_radius_matching_primary"
        ),
        "recommended_next_route_or_none": "8.7.56.5531",
        "selected_followup_route": "pattern_epsilon_bohr_radius_matching",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5529",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "round1_note": sign_base.display_path(ROUND1_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5531",
                "followup_route": "pattern_epsilon_bohr_radius_matching",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_fresh_pattern_round1_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "formulae": build_formulae(),
            "evidence_pack": round1_pack,
        },
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 fresh-pattern Round-1 audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から fresh-pattern Round-1 audit を実行する。

if __name__ == "__main__":
    main()
