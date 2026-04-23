#!/usr/bin/env python3
"""Generate 8.7.56.5555-.5558 Trial-2 scattering / Thomson gate artifacts."""

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
        "8.7.56.5551-5554",
        "updated_pack_trial2_scattering_thomson_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5555-5558"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "scattering / Thomson gate / Ward-reserve refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_scattering_thomson_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_scattering_thomson_target_free_readout_missing_"
    "ward_current_algebra_primary_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_scattering_thomson_negative_closeout_completed_"
    "ward_current_algebra_primary_conditional_reopen_reserve_next"
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
    """Return formulas used by the scattering / Thomson gate."""
    return {
        "gate_a": "Gate A = scattering / Thomson audit available now",
        "gate_b": "Gate B = scattering / Thomson negative closeout completed now",
        "gate_c": "Gate C = Ward / current algebra promoted primary now",
    }


# 関数: `.5555-.5558` を実行する。

def main() -> None:
    """Execute the Trial-2 scattering / Thomson gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(
        prior_summary[
            "exact_trial2_scattering_thomson_selected_extension_scalar_surface_available_now"
        ]
    )
    gate_b = bool(
        prior_summary["exact_trial2_scattering_thomson_negative_closeout_available_now"]
    )
    gate_c = bool(
        prior_summary["updated_pack_trial2_ward_current_algebra_followup_required_now"]
    )
    trial2_scattering_thomson_negative_closeout_completed_now = bool(gate_a and gate_b)
    trial2_ward_current_algebra_primary_next_now = bool(
        trial2_scattering_thomson_negative_closeout_completed_now and gate_c
    )
    conditional_reopen_reserve_retained_now = bool(
        trial2_ward_current_algebra_primary_next_now
    )

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_scattering_thomson_audit_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 scattering / Thomson audit available now",
            sign_base.truth(gate_a),
            "The selected-extension scalar pack and scattering audit verdict are available.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_scattering_thomson_negative_closeout_completed_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 scattering / Thomson negative closeout completed now",
            sign_base.truth(gate_b),
            "The current pack does not materialize one independent low-energy scattering / Thomson alpha readout.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_ward_current_algebra_promoted_primary_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 Ward / current algebra promoted primary now",
            sign_base.truth(gate_c),
            "Once scattering closes negatively, Ward / current algebra becomes the next honest primary route.",
        ),
        sign_base.row(
            "trial2_scattering_thomson_negative_closeout_completed_now",
            "pass"
            if trial2_scattering_thomson_negative_closeout_completed_now
            else "reject",
            "Trial-2 scattering / Thomson negative closeout completed now",
            sign_base.truth(trial2_scattering_thomson_negative_closeout_completed_now),
            "The scattering / Thomson route has now closed honestly under the current pack.",
        ),
        sign_base.row(
            "trial2_ward_current_algebra_primary_next_now",
            "pass" if trial2_ward_current_algebra_primary_next_now else "reject",
            "Trial-2 Ward / current algebra primary next now",
            sign_base.truth(trial2_ward_current_algebra_primary_next_now),
            "The next honest blocker is now the Ward/current-algebra route audit.",
        ),
        sign_base.row(
            "conditional_reopen_reserve_retained_now",
            "pass" if conditional_reopen_reserve_retained_now else "reject",
            "conditional reopen reserve retained now",
            sign_base.truth(conditional_reopen_reserve_retained_now),
            "Conditional reopen stays reserve while the last promoted theorem route, Ward/current algebra, is still live.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "soft_alpha_naive": float(prior_summary["soft_alpha_naive"]),
        "alpha_target": float(prior_summary["alpha_target"]),
        "soft_alpha_target_relative_mismatch": float(
            prior_summary["soft_alpha_target_relative_mismatch"]
        ),
        "best_extra_label_vs_alpha_target": prior_summary[
            "best_extra_label_vs_alpha_target"
        ],
        "best_extra_q_over_m0": float(prior_summary["best_extra_q_over_m0"]),
        "best_extra_alpha_target_residual": float(
            prior_summary["best_extra_alpha_target_residual"]
        ),
        "gate_a_updated_pack_trial2_scattering_thomson_audit_available_now": gate_a,
        "gate_b_updated_pack_trial2_scattering_thomson_negative_closeout_completed_now": gate_b,
        "gate_c_updated_pack_trial2_ward_current_algebra_promoted_primary_now": gate_c,
        "trial2_scattering_thomson_negative_closeout_completed_now": (
            trial2_scattering_thomson_negative_closeout_completed_now
        ),
        "trial2_ward_current_algebra_primary_next_now": (
            trial2_ward_current_algebra_primary_next_now
        ),
        "conditional_reopen_reserve_retained_now": (
            conditional_reopen_reserve_retained_now
        ),
        "selected_primary_completion_lane": "trial2_ward_current_algebra",
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "new_selected_extension_native_source_only",
        "selected_next_generation_route": "trial2_ward_current_algebra",
        "recommended_next_route_or_none": "8.7.56.5559",
        "selected_followup_route": "trial2_ward_current_algebra",
        "selected_followup_route_or_none": "8.7.56.5559",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5557",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5559",
                "followup_route": "conditional_reopen_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_scattering_thomson_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 scattering / Thomson gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から scattering gate を実行する。

if __name__ == "__main__":
    main()
