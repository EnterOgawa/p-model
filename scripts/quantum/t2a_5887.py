#!/usr/bin/env python3
"""Generate 8.7.56.5887-.5890 alpha observable sensitivity inventory artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_alpha_observable_sensitivity_inventory_backend import (
    build_trial2_alpha_observable_sensitivity_inventory_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5883-5886",
        "updated_pack_trial2_4d_self_consistent_selector_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5887-5890"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "alpha observable sensitivity inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_alpha_observable_sensitivity_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_pmodel_native_alpha_observable_comparison_goal_reset_audited_"
    "sensitivity_inventory_primary_independent_filter_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_alpha_observable_sensitivity_inventory_audited_"
    "independent_filter_gate_next"
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


# 関数: sensitivity inventory で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the sensitivity inventory."""
    return {
        "native_value": "alpha_P_frozen := alpha_3D,exact",
        "corrected_layers": "alpha_P_4D,can, alpha_P_4D,vertex are corrected / probe-side layers",
        "comparison_rule": (
            "observable comparison requires explicit alpha dependence, "
            "independence from CODATA-input extraction, and current rerun readiness"
        ),
    }


# 関数: `.5887-.5890` を実行する。

def main() -> None:
    """Execute the alpha observable sensitivity inventory audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_alpha_observable_sensitivity_inventory_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    inventory_complete = bool(pack["trial2_alpha_observable_sensitivity_inventory_complete_now"])
    explicit_alpha_surface_available = bool(summary_pack["explicit_alpha_surface_count"] > 0)
    independent_surface_available = bool(summary_pack["independent_surface_count"] > 0)
    current_primary_ready_unavailable = not bool(
        pack["trial2_alpha_current_primary_ready_surface_available_now"]
    )
    qed_materialization_primary = bool(pack["trial2_alpha_primary_materialization_qed_vacuum_now"])
    weak_materialization_secondary = bool(
        pack["trial2_alpha_secondary_materialization_weak_sector_now"]
    )
    reserve_diagnostic_retained = bool(pack["trial2_alpha_reserve_diagnostic_de_broglie_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_observable_comparison_goal_reset_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 observable-comparison goal-reset selected now",
            sign_base.truth(route_selected),
            "The sensitivity inventory starts only after the exact-goal mainline is demoted and observable comparison becomes the closing strategy.",
        ),
        sign_base.row(
            "trial2_alpha_observable_sensitivity_inventory_complete_now",
            "pass" if inventory_complete else "reject",
            "Trial-2 alpha observable sensitivity inventory complete now",
            sign_base.truth(inventory_complete),
            "The current Part III-B candidate surfaces are now classified by alpha dependency, independence, and rerun readiness.",
        ),
        sign_base.row(
            "trial2_alpha_explicit_surface_available_now",
            "pass" if explicit_alpha_surface_available else "reject",
            "Trial-2 alpha explicit surface available now",
            sign_base.truth(explicit_alpha_surface_available),
            "At least one current script already carries alpha explicitly as a data-carrying object.",
        ),
        sign_base.row(
            "trial2_alpha_independent_surface_available_now",
            "pass" if independent_surface_available else "reject",
            "Trial-2 alpha independent surface available now",
            sign_base.truth(independent_surface_available),
            "The current Part III-B pack does contain independent observables, even when alpha is not yet exposed as a rerun input.",
        ),
        sign_base.row(
            "trial2_alpha_current_primary_ready_surface_unavailable_now",
            "pass" if current_primary_ready_unavailable else "reject",
            "Trial-2 alpha current primary-ready surface unavailable now",
            sign_base.truth(current_primary_ready_unavailable),
            "No current independent surface is simultaneously alpha-explicit and rerun-ready, so the next blocker is filtering and materialization rather than blind rerun.",
        ),
        sign_base.row(
            "trial2_alpha_qed_vacuum_materialization_primary_now",
            "pass" if qed_materialization_primary else "reject",
            "Trial-2 alpha QED-vacuum materialization primary now",
            sign_base.truth(qed_materialization_primary),
            "Casimir / Lamb / H 1S-2S already have retained source packs and high structural alpha sensitivity, so they lead the first materialization route.",
        ),
        sign_base.row(
            "trial2_alpha_weak_sector_materialization_secondary_now",
            "pass" if weak_materialization_secondary else "reject",
            "Trial-2 alpha weak-sector materialization secondary now",
            sign_base.truth(weak_materialization_secondary),
            "The weak-sector pack is independent and still valuable, but alpha is more hidden there than in the QED-vacuum baseline pack.",
        ),
        sign_base.row(
            "trial2_alpha_de_broglie_reserve_diagnostic_retained_now",
            "pass" if reserve_diagnostic_retained else "reject",
            "Trial-2 alpha de Broglie reserve diagnostic retained now",
            sign_base.truth(reserve_diagnostic_retained),
            "The recoil-vs-g-2 cross-check remains useful as a reserve diagnostic, but not as a primary closing target.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "total_candidate_surfaces": int(summary_pack["total_candidate_surfaces"]),
        "explicit_alpha_surface_count": int(summary_pack["explicit_alpha_surface_count"]),
        "codata_overlap_surface_count": int(summary_pack["codata_overlap_surface_count"]),
        "independent_surface_count": int(summary_pack["independent_surface_count"]),
        "primary_ready_surface_count": int(summary_pack["primary_ready_surface_count"]),
        "rerun_ready_independent_surface_count": int(
            summary_pack["rerun_ready_independent_surface_count"]
        ),
        "selected_primary_materialization_observable_id": str(
            summary_pack["selected_primary_materialization_observable_id"]
        ),
        "selected_secondary_materialization_observable_id": str(
            summary_pack["selected_secondary_materialization_observable_id"]
        ),
        "selected_reserve_diagnostic_observable_id": str(
            summary_pack["selected_reserve_diagnostic_observable_id"]
        ),
        "selected_next_generation_route": "trial2_independent_observable_filter_gate",
        "recommended_next_route_or_none": ".5891-.5894",
        "selected_followup_route": "trial2_independent_observable_filter_gate",
        "selected_followup_route_or_none": ".5891-.5894",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5889",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "candidate_rows": {"count": int(summary_pack["total_candidate_surfaces"])},
        },
        rows,
        summary,
        {
            "overall_status": "trial2_alpha_observable_sensitivity_inventory_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "explicit_alpha_surface_count": int(summary_pack["explicit_alpha_surface_count"]),
            "codata_overlap_surface_count": int(summary_pack["codata_overlap_surface_count"]),
            "independent_surface_count": int(summary_pack["independent_surface_count"]),
            "primary_ready_surface_count": int(summary_pack["primary_ready_surface_count"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] inventory gate:", artifacts["json"])


if __name__ == "__main__":
    main()
