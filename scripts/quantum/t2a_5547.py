#!/usr/bin/env python3
"""Generate 8.7.56.5547-.5550 Trial-2 full spectral / Jost gate artifacts."""

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
        "8.7.56.5543-5546",
        "updated_pack_trial2_full_spectral_jost_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5547-5550"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball full spectral / Jost gate"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_full_spectral_jost_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_full_spectral_jost_target_free_selector_missing_scattering_thomson_"
    "primary_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_full_spectral_jost_negative_closeout_completed_scattering_thomson_"
    "primary_ward_current_algebra_reserve_next"
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
    """Return formulas used by the full spectral / Jost gate."""
    return {
        "gate_a": "Gate A = full spectral / Jost operator-level audit available now",
        "gate_b": "Gate B = target-free selector still unavailable now",
        "gate_c": "Gate C = scattering / Thomson-limit promoted primary now",
    }


# 関数: `.5547-.5550` を実行する。

def main() -> None:
    """Execute the Trial-2 full spectral / Jost gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(prior_summary["exact_trial2_full_spectral_jost_operator_available_now"])
    gate_b = not bool(prior_summary["exact_trial2_full_spectral_jost_target_free_selector_available_now"])
    gate_c = bool(prior_summary["updated_pack_trial2_scattering_thomson_primary_followup_required_now"])
    trial2_full_spectral_jost_gate_completed_now = bool(gate_a and gate_b and gate_c)
    trial2_scattering_thomson_primary_next_now = bool(
        trial2_full_spectral_jost_gate_completed_now
    )
    trial2_ward_current_algebra_reserve_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_full_spectral_jost_operator_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 full spectral / Jost operator available now",
            sign_base.truth(gate_a),
            "The route was audited on a genuine operator-level surface rather than a support-band replay.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_full_spectral_jost_target_free_selector_unavailable_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 full spectral / Jost target-free selector unavailable now",
            sign_base.truth(gate_b),
            "Canonical spectral landmarks still fail to select q_exact target-free.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_scattering_thomson_promoted_primary_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 scattering / Thomson promoted primary now",
            sign_base.truth(gate_c),
            "Once Jost closes negatively, scattering / Thomson becomes the next honest primary route.",
        ),
        sign_base.row(
            "trial2_full_spectral_jost_gate_completed_now",
            "pass" if trial2_full_spectral_jost_gate_completed_now else "reject",
            "Trial-2 full spectral / Jost gate completed now",
            sign_base.truth(trial2_full_spectral_jost_gate_completed_now),
            "The full spectral / Jost route is now fixed honestly as a negative closeout.",
        ),
        sign_base.row(
            "trial2_scattering_thomson_primary_next_now",
            "pass" if trial2_scattering_thomson_primary_next_now else "reject",
            "Trial-2 scattering / Thomson primary next now",
            sign_base.truth(trial2_scattering_thomson_primary_next_now),
            "The next honest blocker is now the scattering amplitude / Thomson-limit route audit.",
        ),
        sign_base.row(
            "trial2_ward_current_algebra_reserve_retained_now",
            "pass" if trial2_ward_current_algebra_reserve_retained_now else "reject",
            "Trial-2 Ward / current algebra reserve retained now",
            sign_base.truth(trial2_ward_current_algebra_reserve_retained_now),
            "Ward/current-algebra remains reserve until scattering dead-ends honestly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "born_re_jost_zero_q_over_m0": float(prior_summary["born_re_jost_zero_q_over_m0"]),
        "born_re_jost_zero_rel_error_vs_q_exact": float(
            prior_summary["born_re_jost_zero_rel_error_vs_q_exact"]
        ),
        "exact_phase_peak_q_over_m0": float(prior_summary["exact_phase_peak_q_over_m0"]),
        "exact_phase_peak_rel_error_vs_q_exact": float(
            prior_summary["exact_phase_peak_rel_error_vs_q_exact"]
        ),
        "gate_a_updated_pack_trial2_full_spectral_jost_operator_available_now": gate_a,
        "gate_b_updated_pack_trial2_full_spectral_jost_target_free_selector_unavailable_now": gate_b,
        "gate_c_updated_pack_trial2_scattering_thomson_promoted_primary_now": gate_c,
        "trial2_full_spectral_jost_gate_completed_now": (
            trial2_full_spectral_jost_gate_completed_now
        ),
        "trial2_scattering_thomson_primary_next_now": (
            trial2_scattering_thomson_primary_next_now
        ),
        "trial2_ward_current_algebra_reserve_retained_now": (
            trial2_ward_current_algebra_reserve_retained_now
        ),
        "selected_primary_completion_lane": "trial2_scattering_thomson",
        "selected_secondary_completion_lane": "trial2_ward_current_algebra",
        "selected_reserve_completion_lane": "none_current_pack",
        "selected_next_generation_route": "trial2_scattering_thomson",
        "recommended_next_route_or_none": "8.7.56.5551",
        "selected_followup_route": "trial2_scattering_thomson",
        "selected_followup_route_or_none": "8.7.56.5555",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5549",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5551",
                "followup_route": "8.7.56.5555",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_full_spectral_jost_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 full spectral / Jost gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から full spectral / Jost gate を実行する。

if __name__ == "__main__":
    main()
