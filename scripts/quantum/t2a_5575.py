#!/usr/bin/env python3
"""Generate 8.7.56.5575-.5578 Trial-2 alpha(beta) curve audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_alpha_beta_curve_backend import build_trial2_alpha_beta_curve_pack
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5571-5574",
        "updated_pack_trial2_direct_alpha_self_consistent_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "77_trial2_numeric_alpha_vector_qball_alpha_beta_curve_audit.md"
)

STEP_TAG = "8.7.56.5575-5578"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "alpha(beta) curve audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_alpha_beta_curve_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_direct_alpha_self_consistent_negative_closeout_completed_"
    "alpha_beta_primary_energy_partition_secondary_entropy_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_alpha_beta_family_global_nonunique_local_microshift_"
    "energy_partition_followup_gate_next"
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


# 関数: note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the alpha(beta) audit note carries the expected claims."""
    patterns = (
        "alpha_beta",
        "global unique ではない",
        "microshift",
        "energy partition",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the alpha(beta) curve audit."""
    return {
        "beta_native_scale": "q_star(beta) = (1 - beta^2)^(1/4)",
        "alpha_beta_family": "alpha_beta(beta) = F_beta(q_star(beta))^2 / (4 pi)",
        "root_test": "alpha_beta(beta) = alpha_target",
        "verdict_rule": "global nonunique + local unique microshift => route compresses blocker to beta origin, then energy partition becomes primary",
    }


# 関数: `.5575-.5578` を実行する。

def main() -> None:
    """Execute the Trial-2 alpha(beta) curve audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_alpha_beta_curve_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    alpha_beta_target_crossing_exists_now = bool(pack["alpha_beta_global_root_count"] >= 1)
    alpha_beta_global_unique_now = bool(pack["alpha_beta_global_unique_now"])
    alpha_beta_local_branch_unique_now = bool(pack["alpha_beta_local_branch_unique_now"])
    alpha_beta_exact_route_available_now = bool(pack["alpha_beta_family_exact_route_available_now"])
    alpha_beta_local_microshift_available_now = bool(pack["alpha_beta_local_microshift_available_now"])
    updated_pack_trial2_energy_partition_primary_followup_required_now = bool(
        route_selected
        and note_available
        and alpha_beta_target_crossing_exists_now
        and alpha_beta_local_branch_unique_now
        and alpha_beta_local_microshift_available_now
        and not alpha_beta_exact_route_available_now
        and bool(pack["energy_partition_followup_required_now"])
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_alpha_beta_curve_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 alpha(beta) curve selected now",
            sign_base.truth(route_selected),
            "The direct-alpha fixed-point branch closed negatively, so alpha(beta) is the next honest direct-alpha followup.",
        ),
        sign_base.row(
            "exact_trial2_alpha_beta_curve_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 alpha(beta) curve audit note available now",
            sign_base.truth(note_available),
            "The audit note records the retained alpha(beta) family, the global double root, and the local beta microshift.",
        ),
        sign_base.row(
            "exact_trial2_alpha_beta_target_crossing_exists_now",
            "pass" if alpha_beta_target_crossing_exists_now else "reject",
            "exact Trial-2 alpha(beta) target crossing exists now",
            sign_base.truth(alpha_beta_target_crossing_exists_now),
            "The family is only nontrivial if alpha_beta(beta) reaches the physical alpha target somewhere on the retained localized interval.",
        ),
        sign_base.row(
            "exact_trial2_alpha_beta_global_unique_now",
            "pass" if alpha_beta_global_unique_now else "reject",
            "exact Trial-2 alpha(beta) global unique now",
            sign_base.truth(alpha_beta_global_unique_now),
            "Pass would mean one globally unique beta is selected by alpha_beta(beta) = alpha_target.",
        ),
        sign_base.row(
            "exact_trial2_alpha_beta_local_branch_unique_now",
            "pass" if alpha_beta_local_branch_unique_now else "reject",
            "exact Trial-2 alpha(beta) local branch unique now",
            sign_base.truth(alpha_beta_local_branch_unique_now),
            "The retained high-beta branch is only useful if it contains one local target crossing near the mode-1 beta anchor.",
        ),
        sign_base.row(
            "exact_trial2_alpha_beta_exact_route_available_now",
            "pass" if alpha_beta_exact_route_available_now else "reject",
            "exact Trial-2 alpha(beta) exact route available now",
            sign_base.truth(alpha_beta_exact_route_available_now),
            "Pass would mean the family is globally unique and reproduces the retained mode-1 beta without any extra branch selection.",
        ),
        sign_base.row(
            "exact_trial2_alpha_beta_local_microshift_available_now",
            "pass" if alpha_beta_local_microshift_available_now else "reject",
            "exact Trial-2 alpha(beta) local microshift available now",
            sign_base.truth(alpha_beta_local_microshift_available_now),
            "The useful surviving signal is one local beta microshift near the retained mode-1 branch.",
        ),
        sign_base.row(
            "updated_pack_trial2_energy_partition_primary_followup_required_now",
            "pass" if updated_pack_trial2_energy_partition_primary_followup_required_now else "reject",
            "updated-pack Trial-2 energy-partition primary followup required now",
            sign_base.truth(updated_pack_trial2_energy_partition_primary_followup_required_now),
            "Once alpha(beta) reduces the blocker to one local beta microshift without exact closeout, energy partition becomes the next honest route.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_target": float(pack["alpha_target"]),
        "retained_beta1": float(pack["retained_beta1"]),
        "retained_q_star_over_m0": float(pack["retained_q_star_over_m0"]),
        "retained_alpha_at_q_star": float(pack["retained_alpha_at_q_star"]),
        "retained_alpha_rel_error_vs_target": float(pack["retained_alpha_rel_error_vs_target"]),
        "alpha_beta_global_root_list": pack["alpha_beta_global_root_list"],
        "alpha_beta_global_root_count": int(pack["alpha_beta_global_root_count"]),
        "alpha_beta_global_unique_now": alpha_beta_global_unique_now,
        "alpha_beta_local_branch_root_list": pack["alpha_beta_local_branch_root_list"],
        "alpha_beta_local_branch_unique_now": alpha_beta_local_branch_unique_now,
        "nearest_alpha_beta_root_to_retained": float(pack["nearest_alpha_beta_root_to_retained"]),
        "nearest_alpha_beta_root_rel_shift_vs_retained": float(
            pack["nearest_alpha_beta_root_rel_shift_vs_retained"]
        ),
        "nearest_alpha_beta_root_charge_proxy": float(pack["nearest_alpha_beta_root_charge_proxy"]),
        "nearest_alpha_beta_root_charge_rel_error_vs_retained": float(
            pack["nearest_alpha_beta_root_charge_rel_error_vs_retained"]
        ),
        "nearest_alpha_beta_root_energy_proxy": float(pack["nearest_alpha_beta_root_energy_proxy"]),
        "nearest_alpha_beta_root_energy_rel_error_vs_retained": float(
            pack["nearest_alpha_beta_root_energy_rel_error_vs_retained"]
        ),
        "exact_trial2_alpha_beta_target_crossing_exists_now": alpha_beta_target_crossing_exists_now,
        "exact_trial2_alpha_beta_global_unique_now": alpha_beta_global_unique_now,
        "exact_trial2_alpha_beta_local_branch_unique_now": alpha_beta_local_branch_unique_now,
        "exact_trial2_alpha_beta_exact_route_available_now": alpha_beta_exact_route_available_now,
        "exact_trial2_alpha_beta_local_microshift_available_now": (
            alpha_beta_local_microshift_available_now
        ),
        "updated_pack_trial2_energy_partition_primary_followup_required_now": (
            updated_pack_trial2_energy_partition_primary_followup_required_now
        ),
        "selected_primary_completion_lane": "trial2_energy_partition_ratio",
        "selected_secondary_completion_lane": "trial2_entropy_route",
        "selected_reserve_completion_lane": "trial2_entropy_route",
        "selected_next_generation_route": "trial2_energy_partition_ratio",
        "recommended_next_route_or_none": "8.7.56.5579",
        "selected_followup_route": "trial2_energy_partition_ratio",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5577",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT / "scripts" / "quantum" / "trial2_alpha_beta_curve_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5579",
                "followup_route": "trial2_energy_partition_ratio",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_alpha_beta_curve_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 alpha(beta) curve audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
