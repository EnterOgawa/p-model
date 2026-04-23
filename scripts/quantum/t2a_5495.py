#!/usr/bin/env python3
"""Generate 8.7.56.5495-.5498 Trial-2 blind-overlap theorem audit artifacts."""

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
        "8.7.56.5491-5494",
        "updated_pack_trial2_reopen_route_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
OVERLAP_EVAL = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_numeric_evaluation_metrics.json"
)
SUPPORT_GATE = (
    PUBLIC_OUT
    / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review_declaration_gate_metrics.json"
)
NUMERIC_CLOSE_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5467-5470",
        "updated_pack_trial2_numerical_closeout_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "66_trial2_numeric_alpha_vector_qball_blind_overlap_theorem_audit.md"
)

STEP_TAG = "8.7.56.5495-5498"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "blind-overlap theorem audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_blind_overlap_theorem_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_reopen_route_inventory_audited_blind_overlap_theorem_primary_"
    "spectral_secondary_residue_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_blind_overlap_theorem_target_free_negative_closeout_completed_"
    "spectral_primary_residue_reserve_gate_next"
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


# 関数: blind-overlap theorem audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the blind-overlap theorem audit."""
    return {
        "blind_overlap_functional": (
            "F_blind(q) = int |f0(r)|^2 j0(q r) r^2 dr / int |f0(r)|^2 r^2 dr"
        ),
        "target_matched_root": "F_blind(q_blind) = sqrt(4 pi alpha_target)",
        "practical_law": "alpha_num = F_blind(q_blind)^2 / (4 pi)",
        "theorem_requirement": (
            "O_blind[f0,q] = 0 => q = q_exact without alpha_target, F_target, "
            "or externally chosen exact-scale selector"
        ),
    }


# 関数: note が expected audit claims を含むかを確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the note carries the expected blind-overlap audit."""
    patterns = (
        "practical numerical law",
        "target-free theorem",
        "q_blind = q_exact",
        "strict target-free blind-overlap theorem",
        "spectral distinguished-scale",
    )
    return all(pattern in text for pattern in patterns)


# 関数: `.5495-.5498` を実行する。

def main() -> None:
    """Execute the Trial-2 blind-overlap theorem audit."""
    for path in (PRIOR_GATE, OVERLAP_EVAL, SUPPORT_GATE, NUMERIC_CLOSE_GATE, AUDIT_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    overlap_summary = sign_base.read_json(OVERLAP_EVAL)["summary"]
    support_summary = sign_base.read_json(SUPPORT_GATE)["summary"]
    numeric_close_summary = sign_base.read_json(NUMERIC_CLOSE_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)

    note_available = note_contains_audit(note_text)
    blind_overlap_functional_formula_available_now = bool(
        prior_summary["trial2_blind_overlap_theorem_primary_next_now"]
    )
    blind_overlap_target_matched_numerical_root_available_now = bool(
        overlap_summary["finite_q_target_crossing_exists"]
    )
    blind_overlap_q_blind_machine_matches_q_exact_now = bool(
        abs(
            float(numeric_close_summary["q_blind_over_m0"])
            - float(numeric_close_summary["q_exact_over_m0"])
        )
        <= 1.0e-12
    )
    blind_overlap_practical_numerical_law_available_now = bool(
        blind_overlap_functional_formula_available_now
        and blind_overlap_target_matched_numerical_root_available_now
        and blind_overlap_q_blind_machine_matches_q_exact_now
        and numeric_close_summary[
            "trial2_practical_blind_overlap_numerical_closeout_available_now"
        ]
    )
    blind_overlap_target_free_selector_available_now = bool(
        support_summary["unique_effective_support_scale_available"]
    )
    blind_overlap_target_free_theorem_available_now = bool(
        blind_overlap_practical_numerical_law_available_now
        and blind_overlap_target_free_selector_available_now
        and not numeric_close_summary["trial2_exact_theorem_closeout_still_missing_now"]
    )
    blind_overlap_theorem_lane_negative_closeout_available_now = bool(
        blind_overlap_practical_numerical_law_available_now
        and not blind_overlap_target_free_theorem_available_now
    )
    updated_pack_trial2_spectral_distinguished_scale_primary_followup_required_now = bool(
        blind_overlap_theorem_lane_negative_closeout_available_now
    )

    rows = [
        sign_base.row(
            "exact_trial2_blind_overlap_theorem_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 blind-overlap theorem audit note available now",
            sign_base.truth(note_available),
            "The dedicated audit note exists and separates the practical blind-overlap law from the strict target-free theorem question.",
        ),
        sign_base.row(
            "exact_trial2_blind_overlap_functional_formula_available_now",
            "pass" if blind_overlap_functional_formula_available_now else "reject",
            "exact Trial-2 blind-overlap functional formula available now",
            sign_base.truth(blind_overlap_functional_formula_available_now),
            "The normalized spherical overlap functional itself is a fixed current-pack object.",
        ),
        sign_base.row(
            "exact_trial2_blind_overlap_target_matched_numerical_root_available_now",
            "pass" if blind_overlap_target_matched_numerical_root_available_now else "reject",
            "exact Trial-2 blind-overlap target-matched numerical root available now",
            sign_base.truth(blind_overlap_target_matched_numerical_root_available_now),
            "The old projection-overlap route already fixed one finite-q root by solving F_blind(q)=sqrt(4 pi alpha_target).",
        ),
        sign_base.row(
            "exact_trial2_blind_overlap_q_blind_machine_matches_q_exact_now",
            "pass" if blind_overlap_q_blind_machine_matches_q_exact_now else "reject",
            "exact Trial-2 blind-overlap q_blind machine-matches q_exact now",
            sign_base.truth(blind_overlap_q_blind_machine_matches_q_exact_now),
            "The retained blind overlap crossing and the retained scalar-side crossing coincide to machine precision.",
        ),
        sign_base.row(
            "exact_trial2_blind_overlap_practical_numerical_law_available_now",
            "pass" if blind_overlap_practical_numerical_law_available_now else "reject",
            "exact Trial-2 blind-overlap practical numerical law available now",
            sign_base.truth(blind_overlap_practical_numerical_law_available_now),
            "The current pack already supports the practical law alpha_num = F_blind(q_blind)^2 / (4 pi).",
        ),
        sign_base.row(
            "exact_trial2_blind_overlap_target_free_selector_available_now",
            "pass" if blind_overlap_target_free_selector_available_now else "reject",
            "exact Trial-2 blind-overlap target-free selector available now",
            sign_base.truth(blind_overlap_target_free_selector_available_now),
            "A strict theorem would need one target-free exact-scale selector, but the old support-band review rejected any unique selector under current canon.",
        ),
        sign_base.row(
            "exact_trial2_blind_overlap_target_free_theorem_available_now",
            "pass" if blind_overlap_target_free_theorem_available_now else "reject",
            "exact Trial-2 blind-overlap target-free theorem available now",
            sign_base.truth(blind_overlap_target_free_theorem_available_now),
            "Reject means current blind-overlap data still define q_blind through a target-matched crossing rather than one target-free theorem law.",
        ),
        sign_base.row(
            "exact_trial2_blind_overlap_theorem_lane_negative_closeout_available_now",
            "pass" if blind_overlap_theorem_lane_negative_closeout_available_now else "reject",
            "exact Trial-2 blind-overlap theorem lane negative closeout available now",
            sign_base.truth(blind_overlap_theorem_lane_negative_closeout_available_now),
            "The blind-overlap route closes honestly as a strict-theorem no-go while retaining the already-fixed practical numerical law.",
        ),
        sign_base.row(
            "updated_pack_trial2_spectral_distinguished_scale_primary_followup_required_now",
            "pass"
            if updated_pack_trial2_spectral_distinguished_scale_primary_followup_required_now
            else "reject",
            "updated-pack Trial-2 spectral distinguished-scale primary followup required now",
            sign_base.truth(
                updated_pack_trial2_spectral_distinguished_scale_primary_followup_required_now
            ),
            "Once blind-overlap theoremization closes negatively, the honest next primary route is the spectral distinguished-scale audit.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(numeric_close_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(numeric_close_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(numeric_close_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(numeric_close_summary["delta_q_over_q_star"]),
        "exact_trial2_blind_overlap_theorem_audit_note_available_now": note_available,
        "exact_trial2_blind_overlap_functional_formula_available_now": (
            blind_overlap_functional_formula_available_now
        ),
        "exact_trial2_blind_overlap_target_matched_numerical_root_available_now": (
            blind_overlap_target_matched_numerical_root_available_now
        ),
        "exact_trial2_blind_overlap_q_blind_machine_matches_q_exact_now": (
            blind_overlap_q_blind_machine_matches_q_exact_now
        ),
        "exact_trial2_blind_overlap_practical_numerical_law_available_now": (
            blind_overlap_practical_numerical_law_available_now
        ),
        "exact_trial2_blind_overlap_target_free_selector_available_now": (
            blind_overlap_target_free_selector_available_now
        ),
        "exact_trial2_blind_overlap_target_free_theorem_available_now": (
            blind_overlap_target_free_theorem_available_now
        ),
        "exact_trial2_blind_overlap_theorem_lane_negative_closeout_available_now": (
            blind_overlap_theorem_lane_negative_closeout_available_now
        ),
        "updated_pack_trial2_spectral_distinguished_scale_primary_followup_required_now": (
            updated_pack_trial2_spectral_distinguished_scale_primary_followup_required_now
        ),
        "selected_primary_completion_lane": "spectral_distinguished_scale",
        "selected_secondary_completion_lane": "effective_coupling_residue",
        "selected_reserve_completion_lane": "selected_extension_source_materialization",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "spectral_distinguished_scale_primary"
        ),
        "recommended_next_route_or_none": "8.7.56.5499",
        "selected_followup_route": "spectral_distinguished_scale",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5497",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "overlap_eval": sign_base.display_path(OVERLAP_EVAL),
                "support_gate": sign_base.display_path(SUPPORT_GATE),
                "numeric_close_gate": sign_base.display_path(NUMERIC_CLOSE_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5499",
                "followup_route": "spectral_distinguished_scale",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_blind_overlap_theorem_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 blind-overlap theorem audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から blind-overlap theorem audit を実行する。

if __name__ == "__main__":
    main()
