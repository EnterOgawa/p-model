#!/usr/bin/env python3
"""Generate 8.7.56.5535-.5538 Trial-2 Bohr / Compton matching audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_pattern_epsilon_bohr_radius_matching_backend import (
    build_trial2_pattern_epsilon_bohr_matching_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5531-5534",
        "updated_pack_trial2_fresh_pattern_round1_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "71_trial2_numeric_alpha_vector_qball_bohr_radius_matching_audit.md"
)

STEP_TAG = "8.7.56.5535-5538"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball Bohr radius / Compton matching audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_bohr_radius_matching_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_fresh_pattern_round1_audited_bohr_radius_front_runner_"
    "full_spectral_jost_secondary_scattering_thomson_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_bohr_radius_matching_heuristic_front_runner_target_free_theorem_"
    "missing_full_spectral_jost_secondary_gate_next"
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


# 関数: audit note が expected Bohr/Compton claims を含むか確認する。

def note_contains_bohr_audit(text: str) -> bool:
    """Return whether the Bohr/Compton audit note carries the expected claims."""
    patterns = (
        "Bohr / Compton matching",
        "tail radius",
        "1/8",
        "heuristic front-runner",
        "target-free theorem",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the Bohr / Compton audit."""
    return {
        "bohr_ratio": "alpha_R = alpha_exact * R",
        "tail_radius": "R_tail = 1 / sqrt(epsilon_beta)",
        "integer_family": "alpha_n = sqrt(epsilon_beta) / n",
        "fit_denominator": "n_fit = sqrt(epsilon_beta) / alpha_exact",
    }


# 関数: `.5535-.5538` を実行する。

def main() -> None:
    """Execute the Trial-2 Bohr / Compton matching audit."""
    for path in (PRIOR_GATE, AUDIT_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    audit_note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_pattern_epsilon_bohr_matching_pack()

    note_available = note_contains_bohr_audit(audit_note_text)
    tail_radius_one_eighth_available_now = bool(
        note_available and pack["tail_radius_one_eighth_available_now"]
    )
    one_eighth_integer_front_runner_now = bool(pack["one_eighth_integer_front_runner_now"])
    target_free_theorem_available_now = bool(pack["target_free_theorem_available_now"])
    heuristic_front_runner_only_now = bool(pack["heuristic_front_runner_only_now"])
    negative_closeout_available_now = bool(pack["negative_closeout_available_now"])
    updated_pack_full_spectral_jost_followup_required_now = bool(
        negative_closeout_available_now and not target_free_theorem_available_now
    )

    rows = [
        sign_base.row(
            "exact_trial2_bohr_radius_matching_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 Bohr radius / Compton audit note available now",
            sign_base.truth(note_available),
            "The dedicated note exists and records the canonical-radius and denominator-family screen.",
        ),
        sign_base.row(
            "exact_trial2_bohr_radius_tail_one_eighth_front_runner_available_now",
            "pass" if tail_radius_one_eighth_available_now else "reject",
            "exact Trial-2 Bohr radius tail one-eighth front-runner available now",
            sign_base.truth(tail_radius_one_eighth_available_now),
            "Tail radius is the only canonical radius whose low-complexity ratio lands closest to 1/8.",
        ),
        sign_base.row(
            "exact_trial2_bohr_radius_integer_one_eighth_front_runner_now",
            "pass" if one_eighth_integer_front_runner_now else "reject",
            "exact Trial-2 Bohr radius integer one-eighth front-runner now",
            sign_base.truth(one_eighth_integer_front_runner_now),
            "Among alpha_n = sqrt(epsilon_beta)/n with n=6..12, n=8 is the unique q-space front-runner.",
        ),
        sign_base.row(
            "exact_trial2_bohr_radius_target_free_theorem_available_now",
            "pass" if target_free_theorem_available_now else "reject",
            "exact Trial-2 Bohr radius target-free theorem available now",
            sign_base.truth(target_free_theorem_available_now),
            "Reject means the current pack still lacks a frozen-action identity that selects tail radius and exact denominator 8.",
        ),
        sign_base.row(
            "exact_trial2_bohr_radius_heuristic_front_runner_only_now",
            "pass" if heuristic_front_runner_only_now else "reject",
            "exact Trial-2 Bohr radius heuristic front-runner only now",
            sign_base.truth(heuristic_front_runner_only_now),
            "The route survives only as the best heuristic low-complexity fit, not as a theorem closeout.",
        ),
        sign_base.row(
            "exact_trial2_bohr_radius_negative_closeout_available_now",
            "pass" if negative_closeout_available_now else "reject",
            "exact Trial-2 Bohr radius negative closeout available now",
            sign_base.truth(negative_closeout_available_now),
            "The honest read is that Bohr/Compton matching improves q_star but does not yet close target-free.",
        ),
        sign_base.row(
            "updated_pack_trial2_full_spectral_jost_followup_required_now",
            "pass" if updated_pack_full_spectral_jost_followup_required_now else "reject",
            "updated-pack Trial-2 full spectral / Jost followup required now",
            sign_base.truth(updated_pack_full_spectral_jost_followup_required_now),
            "Once the Bohr route is fixed as heuristic-only, the next honest primary becomes full spectral / Jost.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "q_one_eighth_over_m0": float(pack["q_one_eighth_over_m0"]),
        "q_one_eighth_relative_error_vs_q_exact": float(
            pack["q_one_eighth_relative_error_vs_q_exact"]
        ),
        "alpha_exact_from_q_exact": float(pack["alpha_exact_from_q_exact"]),
        "n_fit_from_exact_ratio": float(pack["n_fit_from_exact_ratio"]),
        "n_fit_relative_gap_vs_8": float(pack["n_fit_relative_gap_vs_8"]),
        "best_radius_label": str(pack["best_radius_label"]),
        "best_radius_fraction_label": str(pack["best_radius_fraction_label"]),
        "best_radius_fraction_relative_gap": float(pack["best_radius_fraction_relative_gap"]),
        "best_integer_denominator": int(pack["best_integer_denominator"]),
        "best_integer_relative_error_vs_q_exact": float(
            pack["best_integer_relative_error_vs_q_exact"]
        ),
        "exact_trial2_bohr_radius_matching_audit_note_available_now": note_available,
        "exact_trial2_bohr_radius_tail_one_eighth_front_runner_available_now": (
            tail_radius_one_eighth_available_now
        ),
        "exact_trial2_bohr_radius_integer_one_eighth_front_runner_now": (
            one_eighth_integer_front_runner_now
        ),
        "exact_trial2_bohr_radius_target_free_theorem_available_now": (
            target_free_theorem_available_now
        ),
        "exact_trial2_bohr_radius_heuristic_front_runner_only_now": (
            heuristic_front_runner_only_now
        ),
        "exact_trial2_bohr_radius_negative_closeout_available_now": (
            negative_closeout_available_now
        ),
        "updated_pack_trial2_full_spectral_jost_followup_required_now": (
            updated_pack_full_spectral_jost_followup_required_now
        ),
        "selected_primary_completion_lane": "trial2_full_spectral_jost",
        "selected_secondary_completion_lane": "trial2_scattering_thomson",
        "selected_reserve_completion_lane": "trial2_ward_current_algebra",
        "selected_next_generation_route": "trial2_full_spectral_jost",
        "recommended_next_route_or_none": "8.7.56.5539",
        "selected_followup_route": "trial2_full_spectral_jost",
        "selected_followup_route_or_none": "8.7.56.5543",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5537",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5539",
                "followup_route": "8.7.56.5543",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_bohr_radius_matching_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae(), "evidence_pack": pack},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 Bohr radius / Compton matching audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から Bohr / Compton audit を実行する。

if __name__ == "__main__":
    main()
