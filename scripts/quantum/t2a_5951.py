#!/usr/bin/env python3
"""Generate 8.7.56.5951-.5954 third-surface inventory refresh artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_third_independent_surface_inventory_refresh_backend import (
    build_trial2_third_independent_surface_inventory_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5947-5950",
        "updated_pack_trial2_hyperfine_attribution_split_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5951-5954"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "third independent surface inventory refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_third_independent_surface_inventory_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_attribution_split_completed_third_surface_inventory_"
    "primary_watch_gate_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_third_independent_surface_unavailable_multi_watch_gate_"
    "primary_conditional_reopen_secondary_next"
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
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# 関数: `.5951-.5954` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the third-surface refresh."""
    return {
        "hydrogen_replay_rule": (
            "Hydrogen Balmer/Lyman lines stay in the same gross-structure alpha^2 "
            "family as 1S-2S and therefore count as replay, not as a genuinely new surface"
        ),
        "helium_rule": (
            "Helium remains observed-only until one deterministic absolute alpha -> "
            "observable formula is materialized"
        ),
        "third_surface_rule": (
            "a third surface is honest only when it is alpha-explicit, rerun-ready, "
            "and genuinely independent of the existing 1S-2S / 21 cm pair"
        ),
    }


# 関数: `.5951-.5954` を実行する。

def main() -> None:
    """Execute the third independent surface inventory refresh gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_third_independent_surface_inventory_refresh_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    replay_candidates = int(summary_pack["hydrogen_gross_replay_candidate_count_now"]) > 0
    replay_not_new = not bool(summary_pack["hydrogen_gross_replay_is_genuinely_new_now"])
    helium_lamb_weak_all_unavailable = (
        not bool(summary_pack["helium_absolute_formula_available_now"])
        and not bool(summary_pack["lamb_absolute_formula_available_now"])
        and not bool(summary_pack["weak_explicit_formula_available_now"])
    )
    genuine_third_unavailable = not bool(summary_pack["genuine_third_independent_surface_available_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_hyperfine_attribution_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 hyperfine attribution selected now",
            sign_base.truth(route_selected),
            "The third-surface inventory refresh starts only after the split origin is localized.",
        ),
        sign_base.row(
            "trial2_hydrogen_gross_replay_candidates_exist_now",
            "pass" if replay_candidates else "reject",
            "Trial-2 hydrogen gross replay candidates exist now",
            sign_base.truth(replay_candidates),
            "Hydrogen Balmer/Lyman lines are replay-capable under the same Coulomb gross-structure family.",
        ),
        sign_base.row(
            "trial2_hydrogen_gross_replay_not_genuinely_new_now",
            "pass" if replay_not_new else "reject",
            "Trial-2 hydrogen gross replay not genuinely new now",
            sign_base.truth(replay_not_new),
            "These additional Hydrogen lines do not count as a third independent family because they replay the same alpha^2 gross-structure law.",
        ),
        sign_base.row(
            "trial2_helium_lamb_weak_third_surface_unavailable_now",
            "pass" if helium_lamb_weak_all_unavailable else "reject",
            "Trial-2 Helium/Lamb/weak third surface unavailable now",
            sign_base.truth(helium_lamb_weak_all_unavailable),
            "No deterministic absolute alpha formula currently materializes on Helium, Lamb, or weak beta decay.",
        ),
        sign_base.row(
            "trial2_genuine_third_surface_unavailable_now",
            "pass" if genuine_third_unavailable else "reject",
            "Trial-2 genuine third surface unavailable now",
            sign_base.truth(genuine_third_unavailable),
            "Current pack still lacks a genuinely new third independent alpha-explicit rerun surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "hydrogen_gross_replay_candidate_count_now": int(summary_pack["hydrogen_gross_replay_candidate_count_now"]),
        "hydrogen_gross_replay_all_codata_best_now": bool(summary_pack["hydrogen_gross_replay_all_codata_best_now"]),
        "helium_absolute_formula_available_now": bool(summary_pack["helium_absolute_formula_available_now"]),
        "lamb_absolute_formula_available_now": bool(summary_pack["lamb_absolute_formula_available_now"]),
        "weak_explicit_formula_available_now": bool(summary_pack["weak_explicit_formula_available_now"]),
        "genuine_third_independent_surface_available_now": bool(
            summary_pack["genuine_third_independent_surface_available_now"]
        ),
        "selected_next_generation_route": "trial2_multi_observable_watch_pass_gate",
        "recommended_next_route_or_none": ".5955-.5958",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": "conditional",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5953",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "hydrogen_replay_rows": pack["hydrogen_replay_rows"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_third_independent_surface_unavailable",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "hydrogen_gross_replay_candidate_count_now": int(
                summary_pack["hydrogen_gross_replay_candidate_count_now"]
            ),
            "genuine_third_independent_surface_available_now": bool(
                summary_pack["genuine_third_independent_surface_available_now"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] third-surface inventory gate:", artifacts["json"])


if __name__ == "__main__":
    main()
