#!/usr/bin/env python3
"""Generate 8.7.56.5567-.5570 Trial-2 direct-alpha self-consistent audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_direct_alpha_self_consistent_route_backend import (
    build_trial2_direct_alpha_self_consistent_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5563-5566",
        "updated_pack_trial2_ward_current_algebra_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "76_trial2_numeric_alpha_vector_qball_direct_alpha_self_consistent_audit.md"
)

STEP_TAG = "8.7.56.5567-5570"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "direct-alpha self-consistent audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_direct_alpha_self_consistent_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_ward_current_algebra_negative_closeout_completed_"
    "conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_direct_alpha_self_consistent_target_free_mismatch_"
    "alpha_beta_followup_gate_next"
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
    """Return whether the direct-alpha audit note carries the expected claims."""
    patterns = (
        "q = \\alpha(q)",
        "q_sc",
        "negative closeout",
        "alpha(beta)",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the direct-alpha self-consistent audit."""
    return {
        "self_consistent_condition": "q = alpha(q) = F(q)^2 / (4 pi)",
        "mismatch_test": "route passes only if q_sc = q_exact up to retained tolerance",
        "followup_rule": "if q_sc exists but q_sc != q_exact, route closes negatively and alpha(beta) becomes the next honest followup",
    }


# 関数: `.5567-.5570` を実行する。

def main() -> None:
    """Execute the Trial-2 direct-alpha self-consistent audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_direct_alpha_self_consistent_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    self_consistent_root_exists_now = bool(pack["self_consistent_root_exists_now"])
    self_consistent_root_unique_now = bool(pack["self_consistent_root_unique_now"])
    target_free_self_consistent_alpha_route_available_now = bool(
        pack["target_free_self_consistent_alpha_route_available_now"]
    )
    target_free_self_consistent_alpha_route_negative_now = bool(
        route_selected
        and note_available
        and self_consistent_root_exists_now
        and self_consistent_root_unique_now
        and not target_free_self_consistent_alpha_route_available_now
        and bool(pack["target_free_self_consistent_alpha_route_negative_now"])
    )
    updated_pack_trial2_alpha_beta_curve_primary_followup_required_now = bool(
        target_free_self_consistent_alpha_route_negative_now
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_direct_alpha_self_consistent_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 direct-alpha self-consistent route selected now",
            sign_base.truth(route_selected),
            "The current pack is on conditional reopen only, so a genuinely new direct-alpha route is an admissible next audit.",
        ),
        sign_base.row(
            "exact_trial2_direct_alpha_self_consistent_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 direct-alpha self-consistent audit note available now",
            sign_base.truth(note_available),
            "The dedicated audit note records the self-consistent condition, the retained root, and the resulting negative closeout.",
        ),
        sign_base.row(
            "exact_trial2_direct_alpha_self_consistent_root_exists_now",
            "pass" if self_consistent_root_exists_now else "reject",
            "exact Trial-2 direct-alpha self-consistent root exists now",
            sign_base.truth(self_consistent_root_exists_now),
            "The equation q = alpha(q) is mathematically nontrivial only if at least one retained finite-q root exists.",
        ),
        sign_base.row(
            "exact_trial2_direct_alpha_self_consistent_root_unique_now",
            "pass" if self_consistent_root_unique_now else "reject",
            "exact Trial-2 direct-alpha self-consistent root unique now",
            sign_base.truth(self_consistent_root_unique_now),
            "A unique retained fixed point is required before the route can claim one target-free scale.",
        ),
        sign_base.row(
            "exact_trial2_direct_alpha_self_consistent_route_available_now",
            "pass" if target_free_self_consistent_alpha_route_available_now else "reject",
            "exact Trial-2 direct-alpha self-consistent route available now",
            sign_base.truth(target_free_self_consistent_alpha_route_available_now),
            "Pass would mean the self-consistent fixed point itself selects q_exact without alpha_target as an input.",
        ),
        sign_base.row(
            "exact_trial2_direct_alpha_self_consistent_route_negative_closeout_available_now",
            "pass" if target_free_self_consistent_alpha_route_negative_now else "reject",
            "exact Trial-2 direct-alpha self-consistent route negative closeout available now",
            sign_base.truth(target_free_self_consistent_alpha_route_negative_now),
            "The route closes negatively because the retained self-consistent fixed point exists but sits far from q_exact and q_blind.",
        ),
        sign_base.row(
            "updated_pack_trial2_alpha_beta_curve_primary_followup_required_now",
            "pass" if updated_pack_trial2_alpha_beta_curve_primary_followup_required_now else "reject",
            "updated-pack Trial-2 alpha(beta) curve primary followup required now",
            sign_base.truth(updated_pack_trial2_alpha_beta_curve_primary_followup_required_now),
            "Once the self-consistent route closes negatively, the next honest direct-alpha followup is the alpha(beta) family rather than another q-selector replay.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(pack["q_exact_over_m0"]),
        "q_blind_over_m0": float(pack["q_blind_over_m0"]),
        "q_star_over_m0": float(pack["q_star_over_m0"]),
        "self_consistent_root_list_over_m0": pack["self_consistent_root_list_over_m0"],
        "primary_q_self_consistent_over_m0": float(pack["primary_q_self_consistent_over_m0"]),
        "alpha_at_q_self_consistent": float(pack["alpha_at_q_self_consistent"]),
        "q_minus_alpha_at_q_exact": float(pack["q_minus_alpha_at_q_exact"]),
        "q_self_consistent_rel_error_vs_q_exact": float(pack["q_self_consistent_rel_error_vs_q_exact"]),
        "q_self_consistent_rel_error_vs_q_star": float(pack["q_self_consistent_rel_error_vs_q_star"]),
        "nearest_root_proximity_label": str(pack["nearest_root_proximity_label"]),
        "nearest_root_proximity_gap_over_m0": float(pack["nearest_root_proximity_gap_over_m0"]),
        "exact_trial2_direct_alpha_self_consistent_root_exists_now": self_consistent_root_exists_now,
        "exact_trial2_direct_alpha_self_consistent_root_unique_now": self_consistent_root_unique_now,
        "exact_trial2_direct_alpha_self_consistent_route_available_now": (
            target_free_self_consistent_alpha_route_available_now
        ),
        "exact_trial2_direct_alpha_self_consistent_route_negative_closeout_available_now": (
            target_free_self_consistent_alpha_route_negative_now
        ),
        "updated_pack_trial2_alpha_beta_curve_primary_followup_required_now": (
            updated_pack_trial2_alpha_beta_curve_primary_followup_required_now
        ),
        "selected_primary_completion_lane": "trial2_alpha_beta_curve",
        "selected_secondary_completion_lane": "trial2_energy_partition_ratio",
        "selected_reserve_completion_lane": "trial2_entropy_route",
        "selected_next_generation_route": "trial2_alpha_beta_curve",
        "recommended_next_route_or_none": "8.7.56.5571",
        "selected_followup_route": "trial2_alpha_beta_curve",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5569",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT
                    / "scripts"
                    / "quantum"
                    / "trial2_direct_alpha_self_consistent_route_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5571",
                "followup_route": "trial2_alpha_beta_curve",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_direct_alpha_self_consistent_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 direct-alpha self-consistent audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
