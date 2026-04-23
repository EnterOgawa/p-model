#!/usr/bin/env python3
"""Generate 8.7.56.5591-.5594 Trial-2 interaction-over-harmonic exact-relation audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_interaction_harmonic_exact_relation_backend import (
    build_trial2_interaction_harmonic_exact_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5587-5590",
        "updated_pack_trial2_energy_partition_ratio_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "79_trial2_numeric_alpha_vector_qball_interaction_harmonic_exact_relation_audit.md"
)

STEP_TAG = "8.7.56.5591-5594"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "interaction-over-harmonic exact relation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_interaction_harmonic_exact_relation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_energy_partition_ratio_audited_interaction_harmonic_front_runner_"
    "entropy_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_harmonic_exact_relation_available_boundary_remainder_"
    "non_negligible_entropy_primary_gate"
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


# 関数: note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the exact-relation audit note carries the expected claims."""
    patterns = (
        "interaction-over-harmonic",
        "boundary term",
        "entropy",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the exact-relation audit."""
    return {
        "front_runner": "R_int_harm = E_int / E_harm",
        "exact_relation": (
            "R_int_harm = epsilon_beta/(1+beta^2) + (1/3) * (E_grad / E_harm) + "
            "((4*pi)/3) * B_vir / E_harm"
        ),
        "verdict": (
            "The exact relation exists, but it does not collapse to one simple "
            "beta-only or beta-plus-gradient law because the boundary remainder "
            "stays non-negligible."
        ),
    }


# 関数: `.5591-.5594` を実行する。

def main() -> None:
    """Execute the Trial-2 interaction-over-harmonic exact-relation audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_interaction_harmonic_exact_pack(
        retained_beta=float(RETAINED_BETA),
        nearest_beta=float(NEAREST_BETA),
    )

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    exact_relation_available_now = bool(pack["exact_relation_available_now"])
    boundary_term_negligible_now = bool(pack["boundary_term_negligible_now"])
    beta_only_collapse_supported_now = bool(pack["beta_only_collapse_supported_now"])
    beta_plus_gradient_collapse_supported_now = bool(
        pack["beta_plus_gradient_collapse_supported_now"]
    )
    interaction_over_harmonic_exact_route_available_now = bool(
        pack["interaction_over_harmonic_exact_route_available_now"]
    )
    interaction_over_harmonic_negative_closeout_available_now = bool(
        pack["interaction_over_harmonic_negative_closeout_available_now"]
    )
    entropy_promoted_primary_now = bool(pack["entropy_promoted_primary_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_interaction_harmonic_exact_relation_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 interaction-over-harmonic exact relation selected now",
            sign_base.truth(route_selected),
            "The energy-partition route compressed the blocker to one screened front runner, so the next honest question is whether that ratio itself closes exactly.",
        ),
        sign_base.row(
            "exact_trial2_interaction_harmonic_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 interaction-over-harmonic audit note available now",
            sign_base.truth(note_available),
            "The audit note records the exact decomposition, the non-negligible boundary remainder, and the entropy handoff.",
        ),
        sign_base.row(
            "exact_trial2_interaction_harmonic_exact_relation_available_now",
            "pass" if exact_relation_available_now else "reject",
            "exact Trial-2 interaction-over-harmonic exact relation available now",
            sign_base.truth(exact_relation_available_now),
            "The retained weighted-EOM / virial identities should reconstruct the screened front runner exactly if the algebra is honest.",
        ),
        sign_base.row(
            "exact_trial2_interaction_harmonic_boundary_term_negligible_now",
            "pass" if boundary_term_negligible_now else "reject",
            "exact Trial-2 interaction-over-harmonic boundary term negligible now",
            sign_base.truth(boundary_term_negligible_now),
            "Pass would mean the exact relation effectively collapses to one simple beta-plus-gradient law without an independent remainder.",
        ),
        sign_base.row(
            "exact_trial2_interaction_harmonic_beta_only_collapse_supported_now",
            "pass" if beta_only_collapse_supported_now else "reject",
            "exact Trial-2 interaction-over-harmonic beta-only collapse supported now",
            sign_base.truth(beta_only_collapse_supported_now),
            "Pass would mean the front runner is already one direct beta law and does not require gradient or boundary data.",
        ),
        sign_base.row(
            "exact_trial2_interaction_harmonic_beta_plus_gradient_collapse_supported_now",
            "pass" if beta_plus_gradient_collapse_supported_now else "reject",
            "exact Trial-2 interaction-over-harmonic beta-plus-gradient collapse supported now",
            sign_base.truth(beta_plus_gradient_collapse_supported_now),
            "Pass would mean the boundary remainder is negligible and the exact relation reduces to beta plus one harmonic correction.",
        ),
        sign_base.row(
            "exact_trial2_interaction_harmonic_exact_route_available_now",
            "pass" if interaction_over_harmonic_exact_route_available_now else "reject",
            "exact Trial-2 interaction-over-harmonic exact route available now",
            sign_base.truth(interaction_over_harmonic_exact_route_available_now),
            "Pass would mean the front runner has become one simple target-free law rather than an exact decomposition with an extra remainder.",
        ),
        sign_base.row(
            "exact_trial2_interaction_harmonic_negative_closeout_available_now",
            "pass" if interaction_over_harmonic_negative_closeout_available_now else "reject",
            "exact Trial-2 interaction-over-harmonic negative closeout available now",
            sign_base.truth(interaction_over_harmonic_negative_closeout_available_now),
            "The exact relation exists, but it still needs an independent boundary remainder, so the front runner cannot be promoted to a one-term target-free theorem.",
        ),
        sign_base.row(
            "trial2_entropy_promoted_primary_now",
            "pass" if entropy_promoted_primary_now else "reject",
            "Trial-2 entropy promoted primary now",
            sign_base.truth(entropy_promoted_primary_now),
            "With the interaction-over-harmonic exact-law route closed negatively, entropy becomes the next honest low-cost direct-alpha branch.",
        ),
    ]

    retained = dict(pack["retained_row"])
    near_row = dict(pack["nearest_row"])
    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_beta1": float(pack["retained_beta1"]),
        "nearest_alpha_beta_root_to_retained": float(
            pack["nearest_alpha_beta_root_to_retained"]
        ),
        "retained_interaction_over_harmonic": float(retained["interaction_over_harmonic"]),
        "retained_beta_term": float(retained["beta_term"]),
        "retained_gradient_term": float(retained["gradient_term"]),
        "retained_boundary_term": float(retained["boundary_term"]),
        "retained_exact_reconstruction_residual": float(
            retained["exact_reconstruction_residual"]
        ),
        "retained_boundary_share_of_front_runner": float(
            retained["boundary_share_of_front_runner"]
        ),
        "retained_gradient_share_of_front_runner": float(
            retained["gradient_share_of_front_runner"]
        ),
        "retained_beta_share_of_front_runner": float(retained["beta_share_of_front_runner"]),
        "retained_beta_only_rel_error_vs_front_runner": float(
            retained["beta_only_residual"] / retained["interaction_over_harmonic"]
        ),
        "retained_beta_plus_gradient_rel_error_vs_front_runner": float(
            retained["beta_plus_gradient_residual"] / retained["interaction_over_harmonic"]
        ),
        "nearest_interaction_over_harmonic": float(near_row["interaction_over_harmonic"]),
        "nearest_boundary_share_of_front_runner": float(
            near_row["boundary_share_of_front_runner"]
        ),
        "nearest_beta_plus_gradient_rel_error_vs_front_runner": float(
            near_row["beta_plus_gradient_residual"] / near_row["interaction_over_harmonic"]
        ),
        "exact_trial2_interaction_harmonic_exact_relation_available_now": (
            exact_relation_available_now
        ),
        "exact_trial2_interaction_harmonic_boundary_term_negligible_now": (
            boundary_term_negligible_now
        ),
        "exact_trial2_interaction_harmonic_beta_only_collapse_supported_now": (
            beta_only_collapse_supported_now
        ),
        "exact_trial2_interaction_harmonic_beta_plus_gradient_collapse_supported_now": (
            beta_plus_gradient_collapse_supported_now
        ),
        "exact_trial2_interaction_harmonic_exact_route_available_now": (
            interaction_over_harmonic_exact_route_available_now
        ),
        "exact_trial2_interaction_harmonic_negative_closeout_available_now": (
            interaction_over_harmonic_negative_closeout_available_now
        ),
        "trial2_entropy_promoted_primary_now": entropy_promoted_primary_now,
        "selected_primary_completion_lane": "trial2_entropy_route",
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "conditional_reopen_only",
        "selected_next_generation_route": "trial2_entropy_route",
        "recommended_next_route_or_none": "8.7.56.5595",
        "selected_followup_route": "trial2_entropy_route",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5593",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT
                    / "scripts"
                    / "quantum"
                    / "trial2_interaction_harmonic_exact_relation_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5595",
                "followup_route": "trial2_entropy_route",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_interaction_harmonic_exact_relation_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 interaction-over-harmonic exact-relation audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
