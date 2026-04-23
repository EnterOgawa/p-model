#!/usr/bin/env python3
"""Generate 8.7.56.5431-.5434 Route-A exact universal twenty-seven audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.scalar_proxy_route_a_eom_perturbation_backend import (
    build_scalar_proxy_route_a_eom_perturbation_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5427-5430",
        "updated_pack_scalar_proxy_route_a_nlo_perturbation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5431-5434"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy "
    "Route-A exact universal twenty-seven derivation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_route_a_exact_universal_twentyseven_derivation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_nlo_universal_twentyseven_response_front_runner_audited_"
    "exact_derivation_primary_route_d_secondary_source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_route_a_exact_universal_twentyseven_no_go_theorem_derived_"
    "route_d_profile_moment_primary_source_materialization_reserve_gate"
)


# Function: write one metrics payload as JSON and CSV.
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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# Function: return formulas used by the exact universal twenty-seven audit.

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the exact universal twenty-seven audit."""
    return {
        "candidate_formula": "C_univ(candidate) = 27 = 3^3",
        "required_fit": "C_univ(required) = q_squared_correction_coeff_fit * g3^2",
        "candidate_gap": (
            "delta_C = C_univ(required) - C_univ(candidate)"
        ),
        "route_a_exact_verdict": (
            "Current Route-A algebra fixes one front-runner candidate C_univ = 27, "
            "but does not yet derive it target-free from the frozen-action EOM; "
            "the exact coefficient theorem remains unavailable."
        ),
    }


# Function: execute `.5431-.5434`.

def main() -> None:
    """Execute the Route-A exact universal twenty-seven derivation audit."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_scalar_proxy_route_a_eom_perturbation_pack()

    route_a_exact_audit_available_now = bool(
        prior_summary["gate_a_updated_pack_exact_scalar_proxy_route_a_nlo_universal_twentyseven_front_runner_available_now"]
        and prior_summary["gate_b_updated_pack_scalar_proxy_route_a_exact_universal_twentyseven_derivation_promoted_next"]
    )
    route_a_exact_universal_twentyseven_candidate_formula_available_now = bool(
        pack["route_a_exact_universal_twentyseven_candidate_formula_available_now"]
    )
    route_a_exact_universal_twentyseven_target_free_derivation_available_now = bool(
        pack["route_a_exact_universal_twentyseven_target_free_derivation_available_now"]
    )
    route_a_exact_universal_twentyseven_no_go_theorem_available_now = bool(
        pack["route_a_exact_universal_twentyseven_no_go_theorem_available_now"]
    )
    route_d_profile_moment_promoted_next_now = bool(
        pack["route_d_profile_moment_promoted_next_now"]
    )
    source_materialization_secondary_reserve_retained_now = True

    rows = [
        sign_base.row(
            "exact_scalar_proxy_route_a_exact_universal_twentyseven_audit_available_now",
            "pass" if route_a_exact_audit_available_now else "reject",
            "exact scalar-proxy Route-A exact universal twenty-seven audit available now",
            sign_base.truth(route_a_exact_audit_available_now),
            "Route A NLO has already reduced the problem to one universal coefficient, so this exact audit is now the honest next theorem-side step.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_exact_universal_twentyseven_candidate_formula_available_now",
            "pass" if route_a_exact_universal_twentyseven_candidate_formula_available_now else "reject",
            "scalar-proxy Route-A exact universal twenty-seven candidate formula available now",
            sign_base.truth(route_a_exact_universal_twentyseven_candidate_formula_available_now),
            "The front-runner candidate C_univ = 27 is now explicit and machine-readable.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_exact_universal_twentyseven_target_free_derivation_available_now",
            "pass" if route_a_exact_universal_twentyseven_target_free_derivation_available_now else "reject",
            "scalar-proxy Route-A exact universal twenty-seven target-free derivation available now",
            sign_base.truth(route_a_exact_universal_twentyseven_target_free_derivation_available_now),
            "This remains false unless the frozen-action EOM alone proves C_univ = 27 without importing q_exact or fit input.",
        ),
        sign_base.row(
            "scalar_proxy_route_a_exact_universal_twentyseven_no_go_theorem_available_now",
            "pass" if route_a_exact_universal_twentyseven_no_go_theorem_available_now else "reject",
            "scalar-proxy Route-A exact universal twenty-seven no-go theorem available now",
            sign_base.truth(route_a_exact_universal_twentyseven_no_go_theorem_available_now),
            "Current Route-A algebra supports 27 as a front-runner candidate but does not yet provide the exact coefficient theorem, so the exact derivation branch closes negatively for now.",
        ),
        sign_base.row(
            "scalar_proxy_route_d_profile_moment_promoted_next_now",
            "pass" if route_d_profile_moment_promoted_next_now else "reject",
            "scalar-proxy Route-D profile moment promoted next now",
            sign_base.truth(route_d_profile_moment_promoted_next_now),
            "With Route A exact coefficient derivation not yet available, the best remaining honest branch is Route D profile moments.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_secondary_reserve_retained_now",
            "pass" if source_materialization_secondary_reserve_retained_now else "reject",
            "selected-extension independent extra-q-range source-materialization secondary reserve retained now",
            sign_base.truth(source_materialization_secondary_reserve_retained_now),
            "Source-materialization stays reserve-only while Route D still provides one honest theorem-side alternative.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta1": float(pack["beta1"]),
        "epsilon_beta": float(pack["epsilon_beta"]),
        "g3_actual": float(pack["g3_actual"]),
        "route_a_nlo_required_universal_q_squared_response_coeff_fit": float(
            pack["route_a_nlo_required_universal_q_squared_response_coeff_fit"]
        ),
        "route_a_nlo_universal_q_squared_response_coeff_candidate": float(
            pack["route_a_nlo_universal_q_squared_response_coeff_candidate"]
        ),
        "route_a_nlo_universal_q_squared_response_coeff_abs_error": float(
            pack["route_a_nlo_universal_q_squared_response_coeff_abs_error"]
        ),
        "route_a_nlo_universal_q_squared_response_coeff_rel_error": float(
            pack["route_a_nlo_universal_q_squared_response_coeff_rel_error"]
        ),
        "route_a_exact_universal_twentyseven_candidate_formula_available_now": (
            route_a_exact_universal_twentyseven_candidate_formula_available_now
        ),
        "route_a_exact_universal_twentyseven_target_free_derivation_available_now": (
            route_a_exact_universal_twentyseven_target_free_derivation_available_now
        ),
        "route_a_exact_universal_twentyseven_no_go_theorem_available_now": (
            route_a_exact_universal_twentyseven_no_go_theorem_available_now
        ),
        "route_d_profile_moment_promoted_next_now": route_d_profile_moment_promoted_next_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_route_d_profile_moment_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_reserve_completion_lane": "updated_pack_scalar_proxy_route_c_virial_audit",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_route_d_profile_moment_audit",
        "recommended_next_route_or_none": "8.7.56.5435",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_route_d_profile_moment_gate",
        "selected_followup_route_or_none": "8.7.56.5439",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5433",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5435",
                "followup_route": "8.7.56.5439",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_route_a_exact_universal_twentyseven_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy Route-A exact universal twenty-seven audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the audit when invoked as one CLI script.

if __name__ == "__main__":
    main()
