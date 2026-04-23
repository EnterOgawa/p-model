#!/usr/bin/env python3
"""Generate 8.7.56.5491-.5494 Trial-2 reopen-route gate artifacts."""

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
        "8.7.56.5487-5490",
        "updated_pack_trial2_reopen_route_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5491-5494"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "reopen-route gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_reopen_route_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_reopen_route_inventory_audited_blind_overlap_theorem_primary_"
    "spectral_secondary_residue_reserve_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_reopen_route_inventory_audited_blind_overlap_theorem_primary_"
    "spectral_secondary_residue_reserve_next"
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


# 関数: reopen-route gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the reopen-route gate."""
    return {
        "gate_a": "Gate A = reopen-route inventory available now",
        "gate_b": "Gate B = blind-overlap theorem route promoted as primary",
        "gate_c": "Gate C = unconditional replay required now",
    }


# 関数: `.5491-.5494` を実行する。

def main() -> None:
    """Execute the Trial-2 reopen-route gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(prior_summary["exact_trial2_reopen_route_inventory_available_now"])
    gate_b = bool(prior_summary["trial2_blind_overlap_theorem_route_promoted_now"])
    gate_c = False
    trial2_reopen_route_inventory_gate_completed_now = bool(gate_a and gate_b)
    trial2_blind_overlap_theorem_primary_next_now = bool(
        trial2_reopen_route_inventory_gate_completed_now
    )
    trial2_spectral_distinguished_scale_secondary_retained_now = True
    trial2_effective_coupling_residue_reserve_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_reopen_route_inventory_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 reopen-route inventory available now",
            sign_base.truth(gate_a),
            "The new reopen-route inventory is available and nonempty.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_blind_overlap_theorem_primary_promoted_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 blind-overlap theorem primary promoted now",
            sign_base.truth(gate_b),
            "The next honest blocker is the blind-overlap theorem route itself.",
        ),
        sign_base.row(
            "gate_c_unconditional_reopen_required_now",
            "reject",
            "gate C unconditional reopen required now",
            0.0,
            "Old exhausted branches still do not justify unconditional replay.",
        ),
        sign_base.row(
            "trial2_reopen_route_inventory_gate_completed_now",
            "pass" if trial2_reopen_route_inventory_gate_completed_now else "reject",
            "Trial-2 reopen-route inventory gate completed now",
            sign_base.truth(trial2_reopen_route_inventory_gate_completed_now),
            "The route-priority promotion is now official and machine-readable.",
        ),
        sign_base.row(
            "trial2_blind_overlap_theorem_primary_next_now",
            "pass" if trial2_blind_overlap_theorem_primary_next_now else "reject",
            "Trial-2 blind-overlap theorem primary next now",
            sign_base.truth(trial2_blind_overlap_theorem_primary_next_now),
            "Blind-overlap theoremization is promoted to the primary followup route.",
        ),
        sign_base.row(
            "trial2_spectral_distinguished_scale_secondary_retained_now",
            "pass" if trial2_spectral_distinguished_scale_secondary_retained_now else "reject",
            "Trial-2 spectral distinguished-scale secondary retained now",
            sign_base.truth(trial2_spectral_distinguished_scale_secondary_retained_now),
            "The spectral distinguished-scale route is retained as secondary rather than replayed immediately.",
        ),
        sign_base.row(
            "trial2_effective_coupling_residue_reserve_retained_now",
            "pass" if trial2_effective_coupling_residue_reserve_retained_now else "reject",
            "Trial-2 effective coupling / residue reserve retained now",
            sign_base.truth(trial2_effective_coupling_residue_reserve_retained_now),
            "The heavier coupling/residue branch remains reserve-only until primary and secondary dead-end.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "gate_a_updated_pack_trial2_reopen_route_inventory_available_now": gate_a,
        "gate_b_updated_pack_trial2_blind_overlap_theorem_primary_promoted_now": gate_b,
        "gate_c_unconditional_reopen_required_now": gate_c,
        "trial2_reopen_route_inventory_gate_completed_now": (
            trial2_reopen_route_inventory_gate_completed_now
        ),
        "trial2_blind_overlap_theorem_primary_next_now": (
            trial2_blind_overlap_theorem_primary_next_now
        ),
        "trial2_spectral_distinguished_scale_secondary_retained_now": (
            trial2_spectral_distinguished_scale_secondary_retained_now
        ),
        "trial2_effective_coupling_residue_reserve_retained_now": (
            trial2_effective_coupling_residue_reserve_retained_now
        ),
        "selected_primary_completion_lane": "blind_overlap_theoremization",
        "selected_secondary_completion_lane": "spectral_distinguished_scale",
        "selected_reserve_completion_lane": "effective_coupling_residue",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "blind_overlap_theorem_primary"
        ),
        "recommended_next_route_or_none": "8.7.56.5495",
        "selected_followup_route": "blind_overlap_theoremization",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5493",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5495",
                "followup_route": "blind_overlap_theoremization",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_reopen_route_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 reopen-route gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から reopen-route gate を実行する。

if __name__ == "__main__":
    main()
