#!/usr/bin/env python3
"""Generate 8.7.56.5987-.5990 non-Hydrogen surface gate refresh artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_non_hydrogen_surface_gate_refresh_backend import (
    build_trial2_non_hydrogen_surface_gate_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5983-5986",
        "updated_pack_trial2_helium_absolute_alpha_formula_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5987-5990"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "non-Hydrogen surface gate refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_non_hydrogen_surface_gate_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_helium_absolute_formula_negative_closeout_completed_"
    "non_hydrogen_gate_primary_hydrogen_only_watch_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_non_hydrogen_surface_unavailable_completed_"
    "hydrogen_only_watch_primary_conditional_reopen_secondary_next"
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


# 関数: `.5987-.5990` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the non-Hydrogen gate refresh."""
    return {
        "non_hydrogen_rule": (
            "count a non-Hydrogen route only when it is deterministic, alpha-explicit, "
            "independent, and rerun-ready"
        ),
        "candidate_rule": (
            "Helium, Lamb, and weak-sector routes remain audited candidates, but "
            "none is actual until an absolute formula materializes"
        ),
        "watch_rule": (
            "retain Hydrogen-only watch while non-Hydrogen actual surface count stays zero"
        ),
    }


# 関数: `.5987-.5990` を実行する。

def main() -> None:
    """Execute the non-Hydrogen surface gate refresh."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_non_hydrogen_surface_gate_refresh_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    hydrogen_retained = int(summary_pack["hydrogen_actual_surface_count_now"]) == 3
    non_hydrogen_unavailable = not bool(summary_pack["non_hydrogen_surface_available_now"])
    hydrogen_only_retained = bool(summary_pack["hydrogen_only_table_retained_now"])

    helium_unavailable = not bool(pack["candidate_rows"][0]["rerun_ready_surface_available_now"])
    lamb_unavailable = not bool(pack["candidate_rows"][1]["rerun_ready_surface_available_now"])
    weak_unavailable = not bool(pack["candidate_rows"][2]["rerun_ready_surface_available_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_helium_negative_closeout_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 Helium negative closeout selected now",
            sign_base.truth(route_selected),
            "The non-Hydrogen gate refresh starts only after the Helium no-go is fixed.",
        ),
        sign_base.row(
            "trial2_three_hydrogen_surfaces_retained_now",
            "pass" if hydrogen_retained else "reject",
            "Trial-2 three Hydrogen surfaces retained now",
            sign_base.truth(hydrogen_retained),
            "The current actual table still consists of the same three Hydrogen surfaces.",
        ),
        sign_base.row(
            "trial2_helium_non_hydrogen_surface_unavailable_now",
            "pass" if helium_unavailable else "reject",
            "Trial-2 Helium non-Hydrogen surface unavailable now",
            sign_base.truth(helium_unavailable),
            "Helium remains observed-only and does not yet enter the actual rerun table.",
        ),
        sign_base.row(
            "trial2_lamb_non_hydrogen_surface_unavailable_now",
            "pass" if lamb_unavailable else "reject",
            "Trial-2 Lamb non-Hydrogen surface unavailable now",
            sign_base.truth(lamb_unavailable),
            "Lamb retains structural alpha sensitivity but still no deterministic absolute rerun formula.",
        ),
        sign_base.row(
            "trial2_weak_non_hydrogen_surface_unavailable_now",
            "pass" if weak_unavailable else "reject",
            "Trial-2 weak non-Hydrogen surface unavailable now",
            sign_base.truth(weak_unavailable),
            "Weak beta-decay still exposes no public fine-structure-alpha rerun surface.",
        ),
        sign_base.row(
            "trial2_non_hydrogen_surface_unavailable_now",
            "pass" if non_hydrogen_unavailable else "reject",
            "Trial-2 non-Hydrogen surface unavailable now",
            sign_base.truth(non_hydrogen_unavailable),
            "No audited non-Hydrogen route currently materializes one actual alpha-explicit rerun-ready surface.",
        ),
        sign_base.row(
            "trial2_hydrogen_only_table_retained_now",
            "pass" if hydrogen_only_retained else "reject",
            "Trial-2 Hydrogen-only table retained now",
            sign_base.truth(hydrogen_only_retained),
            "The current honest observable table remains Hydrogen-only after the non-Hydrogen gate refresh.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "hydrogen_actual_surface_count_now": int(summary_pack["hydrogen_actual_surface_count_now"]),
        "hydrogen_surface_ids_now": list(summary_pack["hydrogen_surface_ids_now"]),
        "non_hydrogen_candidate_route_count_now": int(
            summary_pack["non_hydrogen_candidate_route_count_now"]
        ),
        "non_hydrogen_actual_surface_count_now": int(
            summary_pack["non_hydrogen_actual_surface_count_now"]
        ),
        "non_hydrogen_surface_available_now": bool(
            summary_pack["non_hydrogen_surface_available_now"]
        ),
        "selected_next_generation_route": "trial2_hydrogen_only_watch_gate_refresh",
        "recommended_next_route_or_none": ".5991-.5994",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": "conditional",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5989",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "candidate_rows": pack["candidate_rows"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_non_hydrogen_surface_unavailable_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "hydrogen_actual_surface_count_now": int(summary_pack["hydrogen_actual_surface_count_now"]),
            "non_hydrogen_actual_surface_count_now": int(
                summary_pack["non_hydrogen_actual_surface_count_now"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] non-Hydrogen surface gate:", artifacts["json"])


if __name__ == "__main__":
    main()
