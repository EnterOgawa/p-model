#!/usr/bin/env python3
"""Generate 8.7.56.5983-.5986 Helium absolute-formula audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_helium_absolute_alpha_formula_materialization_backend import (
    build_trial2_helium_absolute_alpha_formula_materialization_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5979-5982",
        "updated_pack_trial2_multi_observable_codata_lead_gate_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5983-5986"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "Helium absolute alpha formula materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_helium_absolute_alpha_formula_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_three_surface_codata_lead_watch_retained_"
    "conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_helium_absolute_formula_negative_closeout_completed_"
    "non_hydrogen_gate_primary_hydrogen_only_watch_secondary_next"
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


# 関数: `.5983-.5986` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the Helium audit."""
    return {
        "helium_rule": (
            "promote Helium only when the public pack carries one deterministic "
            "absolute alpha-to-observable formula on a retained He I surface"
        ),
        "term_rule": (
            "vacuum wavelengths and Aki rows alone do not define one alpha-explicit "
            "rerun surface without explicit upper/lower term mapping"
        ),
        "screening_rule": (
            "multi-electron Helium needs one deterministic screening / effective-charge "
            "law before absolute alpha reruns are honest"
        ),
    }


# 関数: `.5983-.5986` を実行する。

def main() -> None:
    """Execute the Helium absolute alpha-formula materialization audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_helium_absolute_alpha_formula_materialization_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    lines_retained = int(summary_pack["helium_selected_line_count_now"]) > 0
    term_unavailable = not bool(summary_pack["helium_term_assignment_available_now"])
    screening_unavailable = not bool(summary_pack["helium_screening_law_available_now"])
    formula_unavailable = not bool(summary_pack["helium_absolute_formula_available_now"])
    rerun_unavailable = not bool(summary_pack["helium_rerun_ready_surface_available_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_three_surface_codata_lead_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 three-surface CODATA lead selected now",
            sign_base.truth(route_selected),
            "The Helium audit starts only after the three-surface Hydrogen-only CODATA-lead watch is fixed.",
        ),
        sign_base.row(
            "trial2_helium_selected_lines_retained_now",
            "pass" if lines_retained else "reject",
            "Trial-2 Helium selected lines retained now",
            sign_base.truth(lines_retained),
            "The retained He I public cache still exposes observed wavelength rows and transition strengths.",
        ),
        sign_base.row(
            "trial2_helium_term_assignment_unavailable_now",
            "pass" if term_unavailable else "reject",
            "Trial-2 Helium term assignment unavailable now",
            sign_base.truth(term_unavailable),
            "Current selected He I rows do not expose one public upper/lower term map needed for deterministic alpha reruns.",
        ),
        sign_base.row(
            "trial2_helium_screening_law_unavailable_now",
            "pass" if screening_unavailable else "reject",
            "Trial-2 Helium screening law unavailable now",
            sign_base.truth(screening_unavailable),
            "No deterministic screening or effective-charge law is materialized in the current Helium public pack.",
        ),
        sign_base.row(
            "trial2_helium_absolute_formula_unavailable_now",
            "pass" if formula_unavailable else "reject",
            "Trial-2 Helium absolute formula unavailable now",
            sign_base.truth(formula_unavailable),
            "The retained He I pack is observed-only and does not yet carry one explicit alpha-to-observable formula.",
        ),
        sign_base.row(
            "trial2_helium_rerun_surface_unavailable_now",
            "pass" if rerun_unavailable else "reject",
            "Trial-2 Helium rerun surface unavailable now",
            sign_base.truth(rerun_unavailable),
            "Without a deterministic formula, no honest Helium alpha-explicit rerun surface exists in the current pack.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "helium_selected_line_count_now": int(summary_pack["helium_selected_line_count_now"]),
        "helium_selected_line_ids_now": list(summary_pack["helium_selected_line_ids_now"]),
        "helium_term_assignment_available_now": bool(
            summary_pack["helium_term_assignment_available_now"]
        ),
        "helium_screening_law_available_now": bool(
            summary_pack["helium_screening_law_available_now"]
        ),
        "helium_absolute_formula_available_now": bool(
            summary_pack["helium_absolute_formula_available_now"]
        ),
        "helium_rerun_ready_surface_available_now": bool(
            summary_pack["helium_rerun_ready_surface_available_now"]
        ),
        "selected_next_generation_route": "trial2_non_hydrogen_surface_gate_refresh",
        "recommended_next_route_or_none": ".5987-.5990",
        "selected_followup_route": "trial2_hydrogen_only_watch_gate_refresh",
        "selected_followup_route_or_none": ".5991-.5994",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5985",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "helium_lines": pack["helium_lines"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_helium_absolute_formula_negative_closeout_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "helium_selected_line_count_now": int(summary_pack["helium_selected_line_count_now"]),
            "helium_absolute_formula_available_now": bool(
                summary_pack["helium_absolute_formula_available_now"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] helium absolute-formula gate:", artifacts["json"])


if __name__ == "__main__":
    main()
