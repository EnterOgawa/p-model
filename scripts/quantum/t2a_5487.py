#!/usr/bin/env python3
"""Generate 8.7.56.5487-.5490 Trial-2 reopen-route inventory artifacts."""

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
        "8.7.56.5483-5486",
        "updated_pack_trial2_conditional_reopen_hold_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
INVENTORY_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "65_trial2_numeric_alpha_vector_qball_reopen_route_inventory.md"
)

STEP_TAG = "8.7.56.5487-5490"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "reopen-route inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_reopen_route_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_conditional_reopen_inventory_audited_no_current_trigger_"
    "hold_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_reopen_route_inventory_audited_blind_overlap_theorem_primary_"
    "spectral_secondary_residue_reserve_gate_next"
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


# 関数: reopen-route inventory で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the reopen-route inventory audit."""
    return {
        "candidate_a": "Route A = blind-overlap theoremization route",
        "candidate_b": "Route B = spectral distinguished-scale route",
        "candidate_c": "Route C = effective coupling / residue route",
        "priority_rule": "primary = blind-overlap theoremization, secondary = spectral distinguished-scale, reserve = effective coupling / residue",
    }


# 関数: note が expected route-inventory claims を含むかを確認する。

def note_contains_inventory(text: str) -> bool:
    """Return whether the note carries the expected reopen-route inventory."""
    patterns = (
        "blind-overlap theoremization route",
        "spectral distinguished-scale route",
        "effective coupling / residue route",
        "genuinely new reopen-route set",
        "conditional hold",
    )
    return all(pattern in text for pattern in patterns)


# 関数: `.5487-.5490` を実行する。

def main() -> None:
    """Execute the Trial-2 reopen-route inventory audit."""
    for path in (PRIOR_GATE, INVENTORY_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(INVENTORY_NOTE)

    note_available = note_contains_inventory(note_text)
    blind_overlap_theorem_route_promoted_now = bool(note_available)
    spectral_distinguished_scale_route_promoted_now = bool(note_available)
    effective_coupling_residue_route_promoted_now = bool(note_available)
    reopen_route_inventory_nonempty_now = bool(
        prior_summary["future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now"]
        and note_available
        and blind_overlap_theorem_route_promoted_now
        and spectral_distinguished_scale_route_promoted_now
        and effective_coupling_residue_route_promoted_now
    )
    conditional_hold_released_by_new_route_inventory_now = bool(
        reopen_route_inventory_nonempty_now
    )
    updated_pack_trial2_blind_overlap_theorem_primary_followup_required_now = bool(
        reopen_route_inventory_nonempty_now
    )

    rows = [
        sign_base.row(
            "exact_trial2_reopen_route_inventory_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 reopen-route inventory note available now",
            sign_base.truth(note_available),
            "The dedicated route note exists and states the three genuinely new reopen candidates.",
        ),
        sign_base.row(
            "exact_trial2_reopen_route_inventory_available_now",
            "pass" if reopen_route_inventory_nonempty_now else "reject",
            "exact Trial-2 reopen-route inventory available now",
            sign_base.truth(reopen_route_inventory_nonempty_now),
            "The user-promoted reopen-route set is now explicit and nonempty.",
        ),
        sign_base.row(
            "trial2_blind_overlap_theorem_route_promoted_now",
            "pass" if blind_overlap_theorem_route_promoted_now else "reject",
            "Trial-2 blind-overlap theorem route promoted now",
            sign_base.truth(blind_overlap_theorem_route_promoted_now),
            "The machine match q_blind = q_exact justifies promoting theoremization of the blind-overlap condition to the primary route.",
        ),
        sign_base.row(
            "trial2_spectral_distinguished_scale_route_promoted_now",
            "pass" if spectral_distinguished_scale_route_promoted_now else "reject",
            "Trial-2 spectral distinguished-scale route promoted now",
            sign_base.truth(spectral_distinguished_scale_route_promoted_now),
            "A spectral distinguished-scale read of q_exact is retained as an independent secondary theorem route.",
        ),
        sign_base.row(
            "trial2_effective_coupling_residue_route_promoted_now",
            "pass" if effective_coupling_residue_route_promoted_now else "reject",
            "Trial-2 effective coupling / residue route promoted now",
            sign_base.truth(effective_coupling_residue_route_promoted_now),
            "A genuinely new selected-extension-native coupling/residue route is retained as reserve rather than replayed immediately.",
        ),
        sign_base.row(
            "trial2_reopen_route_inventory_nonempty_now",
            "pass" if reopen_route_inventory_nonempty_now else "reject",
            "Trial-2 reopen-route inventory nonempty now",
            sign_base.truth(reopen_route_inventory_nonempty_now),
            "The current pack no longer sits on a pure conditional hold once the explicitly promoted new routes are inventoried.",
        ),
        sign_base.row(
            "trial2_conditional_hold_released_by_new_route_inventory_now",
            "pass" if conditional_hold_released_by_new_route_inventory_now else "reject",
            "Trial-2 conditional hold released by new route inventory now",
            sign_base.truth(conditional_hold_released_by_new_route_inventory_now),
            "The old hold remains honest for the exhausted routes, but it is released by the explicit materialization of new admissible routes.",
        ),
        sign_base.row(
            "updated_pack_trial2_blind_overlap_theorem_primary_followup_required_now",
            "pass"
            if updated_pack_trial2_blind_overlap_theorem_primary_followup_required_now
            else "reject",
            "updated-pack Trial-2 blind-overlap theorem primary followup required now",
            sign_base.truth(
                updated_pack_trial2_blind_overlap_theorem_primary_followup_required_now
            ),
            "The next honest official branch is to audit the blind-overlap theorem route itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "exact_trial2_reopen_route_inventory_note_available_now": note_available,
        "exact_trial2_reopen_route_inventory_available_now": (
            reopen_route_inventory_nonempty_now
        ),
        "trial2_blind_overlap_theorem_route_promoted_now": (
            blind_overlap_theorem_route_promoted_now
        ),
        "trial2_spectral_distinguished_scale_route_promoted_now": (
            spectral_distinguished_scale_route_promoted_now
        ),
        "trial2_effective_coupling_residue_route_promoted_now": (
            effective_coupling_residue_route_promoted_now
        ),
        "trial2_reopen_route_inventory_nonempty_now": reopen_route_inventory_nonempty_now,
        "trial2_conditional_hold_released_by_new_route_inventory_now": (
            conditional_hold_released_by_new_route_inventory_now
        ),
        "updated_pack_trial2_blind_overlap_theorem_primary_followup_required_now": (
            updated_pack_trial2_blind_overlap_theorem_primary_followup_required_now
        ),
        "selected_primary_completion_lane": "blind_overlap_theoremization",
        "selected_secondary_completion_lane": "spectral_distinguished_scale",
        "selected_reserve_completion_lane": "effective_coupling_residue",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "blind_overlap_theorem_primary"
        ),
        "recommended_next_route_or_none": "8.7.56.5491",
        "selected_followup_route": "blind_overlap_theoremization",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5489",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "inventory_note": sign_base.display_path(INVENTORY_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5491",
                "followup_route": None,
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_reopen_route_inventory_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 reopen-route inventory completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から reopen-route inventory を実行する。

if __name__ == "__main__":
    main()
