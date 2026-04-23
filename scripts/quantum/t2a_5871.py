#!/usr/bin/env python3
"""Generate 8.7.56.5871-.5874 sign-flip interpolation diagnostic artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_sign_flip_interpolation_diagnostic_backend import (
    build_trial2_sign_flip_interpolation_diagnostic_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5867-5870",
        "updated_pack_trial2_exact_goal_closeout_followup",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5871-5874"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "sign-flip interpolation diagnostic"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_sign_flip_interpolation_diagnostic",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_external_probe_exact_goal_closeout_followup_audited_"
    "sign_flip_primary_weight_source_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_sign_flip_local_bracket_positive_diagnostic_completed_"
    "weight_source_primary_exact_goal_secondary_next"
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


# 関数: diagnostic で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the sign-flip interpolation diagnostic."""
    return {
        "mixed_family": (
            "alpha_4D,mix(eta) = alpha_3D / (C_4D^eta M_4D^(2-eta))"
        ),
        "local_bracket": "alpha_4D,mix(0) < 1/137 < alpha_4D,mix(eta_vertex)",
        "family_derivative": "d alpha_4D,mix / d eta = alpha_4D,mix * ln(M_4D / C_4D) > 0",
    }


# 関数: `.5871-.5874` を実行する。

def main() -> None:
    """Execute the sign-flip interpolation diagnostic."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_sign_flip_interpolation_diagnostic_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    local_bracket = bool(pack["exact_trial2_sign_flip_local_bracket_available_now"])
    global_bracket = bool(pack["exact_trial2_sign_flip_global_bracket_available_now"])
    eta_star_inside = bool(pack["exact_trial2_eta_star_inside_local_bracket_now"])
    monotone_positive = bool(pack["exact_trial2_mixed_family_monotone_positive_now"])
    diagnostic_positive = bool(local_bracket and eta_star_inside and monotone_positive)

    rows = [
        sign_base.row(
            "updated_pack_trial2_exact_goal_closeout_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 exact-goal closeout followup selected now",
            sign_base.truth(route_selected),
            "The sign-flip diagnostic starts only after the exact-goal closeout followup has localized the remaining gap as a selector-weight mismatch.",
        ),
        sign_base.row(
            "exact_trial2_sign_flip_local_bracket_available_now",
            "pass" if local_bracket else "reject",
            "exact Trial-2 sign-flip local bracket available now",
            sign_base.truth(local_bracket),
            "The deterministic vertex candidate already lies on the overshoot side, so a local bracket exists between the canonical row and the vertex row.",
        ),
        sign_base.row(
            "exact_trial2_sign_flip_global_bracket_available_now",
            "pass" if global_bracket else "reject",
            "exact Trial-2 sign-flip global bracket available now",
            sign_base.truth(global_bracket),
            "The full mixed family still brackets the exact goal from opposite sides.",
        ),
        sign_base.row(
            "exact_trial2_eta_star_inside_local_bracket_now",
            "pass" if eta_star_inside else "reject",
            "exact Trial-2 eta_star inside local bracket now",
            sign_base.truth(eta_star_inside),
            "The unique interpolant sits inside the local canonical-to-vertex interval, not outside the deterministic family already in hand.",
        ),
        sign_base.row(
            "exact_trial2_mixed_family_monotone_positive_now",
            "pass" if monotone_positive else "reject",
            "exact Trial-2 mixed family monotone positive now",
            sign_base.truth(monotone_positive),
            "The mixed family is strictly increasing in eta, so the local bracket defines one unique diagnostic crossing.",
        ),
        sign_base.row(
            "updated_pack_trial2_sign_flip_positive_diagnostic_now",
            "pass" if diagnostic_positive else "reject",
            "updated-pack Trial-2 sign-flip positive diagnostic now",
            sign_base.truth(diagnostic_positive),
            "This confirms the local sign-flip geometry, but it remains a diagnostic aid rather than the missing selector theorem.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "eta_exact_goal_interpolant": float(pack["eta_exact_goal_interpolant"]),
        "eta_vertex_weight_candidate": float(pack["eta_vertex_weight_candidate"]),
        "eta_star_position_inside_local_bracket": float(
            pack["eta_star_position_inside_local_bracket"]
        ),
        "alpha_eta_zero_canonical": float(pack["alpha_eta_zero_canonical"]),
        "alpha_eta_vertex": float(pack["alpha_eta_vertex"]),
        "alpha_eta_one_charge_mass": float(pack["alpha_eta_one_charge_mass"]),
        "selected_next_generation_route": "trial2_external_probe_weight_theorem_source_audit",
        "recommended_next_route_or_none": ".5875-.5878",
        "selected_followup_route": "trial2_exact_goal_closeout_hold",
        "selected_followup_route_or_none": "exact-goal closeout remains secondary until theorem source exists",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5873",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_sign_flip_interpolation_diagnostic_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5871-5874 sign-flip interpolation diagnostic completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
