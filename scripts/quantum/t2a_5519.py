#!/usr/bin/env python3
"""Generate 8.7.56.5519-.5522 Trial-2 new-route inventory artifacts."""

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
        "8.7.56.5515-5518",
        "updated_pack_trial2_effective_coupling_residue_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
INVENTORY_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "69_trial2_numeric_alpha_vector_qball_new_route_inventory.md"
)

STEP_TAG = "8.7.56.5519-5522"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "new-route inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_new_route_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_effective_coupling_residue_negative_closeout_completed_"
    "reopen_route_inventory_exhausted_conditional_hold_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_new_route_inventory_audited_full_spectral_jost_primary_"
    "scattering_thomson_secondary_ward_current_algebra_reserve_gate_next"
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


# 関数: new-route inventory で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the Trial-2 new-route inventory audit."""
    return {
        "candidate_a": "Route A = full spectral / Jost route",
        "candidate_b": "Route B = scattering amplitude / Thomson-limit route",
        "candidate_c": "Route C = Ward identity / current algebra route",
        "priority_rule": "primary = full spectral / Jost, secondary = scattering amplitude / Thomson-limit, reserve = Ward identity / current algebra",
    }


# 関数: note が expected new-route claims を含むかを確認する。

def note_contains_inventory(text: str) -> bool:
    """Return whether the note carries the expected new-route inventory."""
    patterns = (
        "full spectral / Jost route",
        "scattering amplitude / Thomson-limit route",
        "Ward identity / current algebra route",
        "genuinely new route set",
        "conditional hold",
    )
    return all(pattern in text for pattern in patterns)


# 関数: `.5519-.5522` を実行する。

def main() -> None:
    """Execute the Trial-2 new-route inventory audit."""
    for path in (PRIOR_GATE, INVENTORY_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(INVENTORY_NOTE)

    note_available = note_contains_inventory(note_text)
    full_spectral_jost_route_promoted_now = bool(note_available)
    scattering_thomson_route_promoted_now = bool(note_available)
    ward_current_algebra_route_promoted_now = bool(note_available)
    new_route_inventory_nonempty_now = bool(
        prior_summary["trial2_reopen_route_inventory_exhausted_now"]
        and note_available
        and full_spectral_jost_route_promoted_now
        and scattering_thomson_route_promoted_now
        and ward_current_algebra_route_promoted_now
    )
    conditional_hold_released_by_new_route_inventory_now = bool(
        new_route_inventory_nonempty_now
    )
    updated_pack_trial2_full_spectral_jost_primary_followup_required_now = bool(
        new_route_inventory_nonempty_now
    )

    rows = [
        sign_base.row(
            "exact_trial2_new_route_inventory_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 new-route inventory note available now",
            sign_base.truth(note_available),
            "The dedicated route note exists and states the three genuinely new routes.",
        ),
        sign_base.row(
            "exact_trial2_new_route_inventory_available_now",
            "pass" if new_route_inventory_nonempty_now else "reject",
            "exact Trial-2 new-route inventory available now",
            sign_base.truth(new_route_inventory_nonempty_now),
            "The user-promoted new-route set is now explicit and nonempty.",
        ),
        sign_base.row(
            "trial2_full_spectral_jost_route_promoted_now",
            "pass" if full_spectral_jost_route_promoted_now else "reject",
            "Trial-2 full spectral / Jost route promoted now",
            sign_base.truth(full_spectral_jost_route_promoted_now),
            "The primary reopen route now asks whether a Jost/resolvent spectral object selects q_exact target-free.",
        ),
        sign_base.row(
            "trial2_scattering_thomson_route_promoted_now",
            "pass" if scattering_thomson_route_promoted_now else "reject",
            "Trial-2 scattering amplitude / Thomson-limit route promoted now",
            sign_base.truth(scattering_thomson_route_promoted_now),
            "A low-energy scattering amplitude readout is retained as an independent secondary computation route.",
        ),
        sign_base.row(
            "trial2_ward_current_algebra_route_promoted_now",
            "pass" if ward_current_algebra_route_promoted_now else "reject",
            "Trial-2 Ward identity / current algebra route promoted now",
            sign_base.truth(ward_current_algebra_route_promoted_now),
            "A conserved-current normalization route is retained as reserve rather than replayed immediately.",
        ),
        sign_base.row(
            "trial2_new_route_inventory_nonempty_now",
            "pass" if new_route_inventory_nonempty_now else "reject",
            "Trial-2 new-route inventory nonempty now",
            sign_base.truth(new_route_inventory_nonempty_now),
            "The current pack no longer sits on a pure hold once the explicitly promoted new routes are inventoried.",
        ),
        sign_base.row(
            "trial2_conditional_hold_released_by_new_route_inventory_now",
            "pass" if conditional_hold_released_by_new_route_inventory_now else "reject",
            "Trial-2 conditional hold released by new-route inventory now",
            sign_base.truth(conditional_hold_released_by_new_route_inventory_now),
            "The old hold remains honest for exhausted routes, but it is released by the explicit materialization of new admissible routes.",
        ),
        sign_base.row(
            "updated_pack_trial2_full_spectral_jost_primary_followup_required_now",
            "pass"
            if updated_pack_trial2_full_spectral_jost_primary_followup_required_now
            else "reject",
            "updated-pack Trial-2 full spectral / Jost primary followup required now",
            sign_base.truth(
                updated_pack_trial2_full_spectral_jost_primary_followup_required_now
            ),
            "The next honest official branch is to audit the full spectral / Jost route itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "exact_trial2_new_route_inventory_note_available_now": note_available,
        "exact_trial2_new_route_inventory_available_now": (
            new_route_inventory_nonempty_now
        ),
        "trial2_full_spectral_jost_route_promoted_now": (
            full_spectral_jost_route_promoted_now
        ),
        "trial2_scattering_thomson_route_promoted_now": (
            scattering_thomson_route_promoted_now
        ),
        "trial2_ward_current_algebra_route_promoted_now": (
            ward_current_algebra_route_promoted_now
        ),
        "trial2_new_route_inventory_nonempty_now": new_route_inventory_nonempty_now,
        "trial2_conditional_hold_released_by_new_route_inventory_now": (
            conditional_hold_released_by_new_route_inventory_now
        ),
        "updated_pack_trial2_full_spectral_jost_primary_followup_required_now": (
            updated_pack_trial2_full_spectral_jost_primary_followup_required_now
        ),
        "selected_primary_completion_lane": "full_spectral_jost",
        "selected_secondary_completion_lane": "scattering_thomson_limit",
        "selected_reserve_completion_lane": "ward_current_algebra",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "full_spectral_jost_primary"
        ),
        "recommended_next_route_or_none": "8.7.56.5523",
        "selected_followup_route": "full_spectral_jost",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5521",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "inventory_note": sign_base.display_path(INVENTORY_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5523",
                "followup_route": None,
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_new_route_inventory_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 new-route inventory audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から new-route inventory audit を実行する。

if __name__ == "__main__":
    main()
