#!/usr/bin/env python3
"""Generate 8.7.56.5515-.5518 Trial-2 effective coupling / residue gate artifacts."""

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
        "8.7.56.5511-5514",
        "updated_pack_trial2_effective_coupling_residue_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5515-5518"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "effective coupling / residue gate / source-materialization-secondary refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_effective_coupling_residue_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_effective_coupling_residue_current_pack_readout_negative_closeout_"
    "completed_conditional_hold_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_effective_coupling_residue_negative_closeout_completed_"
    "reopen_route_inventory_exhausted_conditional_hold_next"
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


# 関数: gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the effective coupling / residue gate."""
    return {
        "gate_a": "Gate A = effective coupling / residue audit available now",
        "gate_b": "Gate B = effective coupling / residue negative closeout completed now",
        "gate_c": "Gate C = selected-extension source-materialization secondary refresh required now",
    }


# 関数: `.5515-.5518` を実行する。

def main() -> None:
    """Execute the Trial-2 effective coupling / residue gate / hold refresh."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_effective_coupling_residue_backend_surface_available_now"]
    )
    gate_b = bool(
        prior_summary[
            "exact_trial2_effective_coupling_residue_lane_negative_closeout_available_now"
        ]
    )
    gate_c = False
    trial2_effective_coupling_residue_negative_closeout_completed_now = bool(
        gate_a and gate_b
    )
    trial2_reopen_route_inventory_exhausted_now = bool(
        trial2_effective_coupling_residue_negative_closeout_completed_now
    )
    trial2_conditional_hold_restored_now = bool(
        trial2_reopen_route_inventory_exhausted_now
    )
    no_unconditional_next_route_now = bool(trial2_conditional_hold_restored_now)
    future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_trial2_effective_coupling_residue_audit_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact Trial-2 effective coupling / residue audit available now",
            sign_base.truth(gate_a),
            "The residue-route audit note and machine-readable verdict are available.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_effective_coupling_residue_negative_closeout_completed_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 effective coupling / residue negative closeout completed now",
            sign_base.truth(gate_b),
            "The current pack exposes no independent current-vertex / pole-residue readout beyond the blind replay surface.",
        ),
        sign_base.row(
            "gate_c_updated_pack_selected_extension_source_materialization_secondary_refresh_required_now",
            "reject",
            "gate C updated-pack selected-extension source-materialization secondary refresh required now",
            0.0,
            "Source-materialization was already exhausted earlier and does not become live again after the residue no-go.",
        ),
        sign_base.row(
            "trial2_effective_coupling_residue_negative_closeout_completed_now",
            "pass"
            if trial2_effective_coupling_residue_negative_closeout_completed_now
            else "reject",
            "Trial-2 effective coupling / residue negative closeout completed now",
            sign_base.truth(
                trial2_effective_coupling_residue_negative_closeout_completed_now
            ),
            "The last promoted reopen route is now honestly closed under the current pack.",
        ),
        sign_base.row(
            "trial2_reopen_route_inventory_exhausted_now",
            "pass" if trial2_reopen_route_inventory_exhausted_now else "reject",
            "Trial-2 reopen-route inventory exhausted now",
            sign_base.truth(trial2_reopen_route_inventory_exhausted_now),
            "Blind-overlap theorem, spectral distinguished-scale, and effective coupling / residue routes are all exhausted under the current pack.",
        ),
        sign_base.row(
            "trial2_conditional_hold_restored_now",
            "pass" if trial2_conditional_hold_restored_now else "reject",
            "Trial-2 conditional hold restored now",
            sign_base.truth(trial2_conditional_hold_restored_now),
            "The honest state returns to conditional hold because no promoted reopen route remains live.",
        ),
        sign_base.row(
            "no_unconditional_next_route_now",
            "pass" if no_unconditional_next_route_now else "reject",
            "no unconditional next route now",
            sign_base.truth(no_unconditional_next_route_now),
            "There is again no unconditional current-pack branch after the residue route closes negatively.",
        ),
        sign_base.row(
            "future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now",
            "pass"
            if future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now
            else "reject",
            "future reopen requires new target-free theorem route or new independent source now",
            sign_base.truth(
                future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now
            ),
            "Only genuinely new theorem or selected-extension-native source/computation branches can justify reopening from here.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "gate_a_updated_pack_exact_trial2_effective_coupling_residue_audit_available_now": gate_a,
        "gate_b_updated_pack_trial2_effective_coupling_residue_negative_closeout_completed_now": gate_b,
        "gate_c_updated_pack_selected_extension_source_materialization_secondary_refresh_required_now": gate_c,
        "trial2_effective_coupling_residue_negative_closeout_completed_now": (
            trial2_effective_coupling_residue_negative_closeout_completed_now
        ),
        "trial2_reopen_route_inventory_exhausted_now": (
            trial2_reopen_route_inventory_exhausted_now
        ),
        "trial2_conditional_hold_restored_now": trial2_conditional_hold_restored_now,
        "no_unconditional_next_route_now": no_unconditional_next_route_now,
        "future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now": (
            future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now
        ),
        "selected_primary_completion_lane": "conditional_reopen_only",
        "selected_secondary_completion_lane": (
            "new_target_free_theorem_route_or_new_independent_source"
        ),
        "selected_reserve_completion_lane": "selected_extension_source_materialization",
        "selected_next_generation_route": None,
        "recommended_next_route_or_none": None,
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5517",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": None,
                "followup_route": "conditional_reopen_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_effective_coupling_residue_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 effective coupling / residue gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から effective coupling / residue gate を実行する。

if __name__ == "__main__":
    main()
