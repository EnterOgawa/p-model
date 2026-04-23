#!/usr/bin/env python3
"""Generate 8.7.56.5559-.5562 Trial-2 Ward identity / current algebra audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_ward_current_algebra_route_backend import (
    build_trial2_ward_current_algebra_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5555-5558",
        "updated_pack_trial2_scattering_thomson_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "74_trial2_numeric_alpha_vector_qball_ward_current_algebra_audit.md"
)

STEP_TAG = "8.7.56.5559-5562"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "Ward identity / current algebra audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_ward_current_algebra_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_scattering_thomson_negative_closeout_completed_"
    "ward_current_algebra_primary_conditional_reopen_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_ward_current_algebra_target_free_readout_missing_"
    "conditional_reopen_gate_next"
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


# 関数: audit note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the dedicated Ward/current-algebra note carries the claims."""
    patterns = (
        "Ward identity / current algebra route",
        "J_Noether^mu[Q]",
        "J_eff^mu[a;Q]",
        "F(0)=1",
        "conditional reopen",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the Ward/current-algebra audit."""
    return {
        "required_surface": (
            "S_Ward := {J_sel^mu(x), Gamma_sel^mu(q), q_mu Gamma_sel^mu(q)=0, "
            "[Q_sel, J_sel^0], alpha_Ward}"
        ),
        "retained_background_current": (
            "J_Noether^mu[Q] = 2 (partial^mu theta) (-Q_g^2)"
        ),
        "retained_source_no_go": (
            "same-field source theorem: J_eff^mu[a;Q] = 0 on the current pack"
        ),
        "collapse_rule": "alpha_soft,naive := F_blind(0)^2 / (4 pi) = 1 / (4 pi)",
    }


# 関数: `.5559-.5562` を実行する。

def main() -> None:
    """Execute the Trial-2 Ward/current-algebra audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_ward_current_algebra_pack()

    route_selected = bool(prior_summary["trial2_ward_current_algebra_primary_next_now"])
    note_available = note_contains_audit(note_text)
    background_noether_current_available_now = bool(
        pack["background_noether_current_available_now"]
    )
    same_field_source_no_go_available_now = bool(
        pack["same_field_source_no_go_available_now"]
    )
    selected_extension_scalar_surface_available_now = bool(
        pack["selected_extension_scalar_surface_available_now"]
    )
    selected_extension_current_surface_available_now = bool(
        pack["selected_extension_current_surface_available_now"]
    )
    selected_extension_ward_identity_surface_available_now = bool(
        pack["selected_extension_ward_identity_surface_available_now"]
    )
    selected_extension_current_algebra_surface_available_now = bool(
        pack["selected_extension_current_algebra_surface_available_now"]
    )
    naive_soft_limit_charge_normalization_collapse_theorem_available_now = bool(
        pack["naive_soft_limit_charge_normalization_collapse_now"]
    )
    independent_ward_alpha_readout_available_now = bool(
        pack["independent_ward_alpha_readout_available_now"]
    )
    exact_trial2_ward_current_algebra_negative_closeout_available_now = bool(
        route_selected
        and note_available
        and background_noether_current_available_now
        and same_field_source_no_go_available_now
        and selected_extension_scalar_surface_available_now
        and not selected_extension_current_surface_available_now
        and not selected_extension_ward_identity_surface_available_now
        and not selected_extension_current_algebra_surface_available_now
        and naive_soft_limit_charge_normalization_collapse_theorem_available_now
        and not independent_ward_alpha_readout_available_now
    )
    updated_pack_trial2_conditional_reopen_refresh_required_now = bool(
        exact_trial2_ward_current_algebra_negative_closeout_available_now
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_ward_current_algebra_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 Ward / current algebra route selected now",
            sign_base.truth(route_selected),
            "The route is worth auditing only after scattering / Thomson has closed negatively and Ward/current algebra has been promoted to primary.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 Ward / current algebra audit note available now",
            sign_base.truth(note_available),
            "The dedicated Ward/current-algebra note exists and records the current-pack object split and no-go logic.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_background_noether_current_available_now",
            "pass" if background_noether_current_available_now else "reject",
            "exact Trial-2 Ward / current algebra background Noether current available now",
            sign_base.truth(background_noether_current_available_now),
            "The older updated-pack branch already derived the conserved background Noether current J_Noether^mu[Q].",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_same_field_source_no_go_available_now",
            "pass" if same_field_source_no_go_available_now else "reject",
            "exact Trial-2 Ward / current algebra same-field source no-go available now",
            sign_base.truth(same_field_source_no_go_available_now),
            "The older exact source theorem already fixed that the same-field photon-side source remains zero on the current pack.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_selected_extension_scalar_surface_available_now",
            "pass" if selected_extension_scalar_surface_available_now else "reject",
            "exact Trial-2 Ward / current algebra selected-extension scalar surface available now",
            sign_base.truth(selected_extension_scalar_surface_available_now),
            "The current selected-extension pack does materialize retained-q and helper-backed extra-q scalar checkpoints; the question is whether it also materializes Ward/current-algebra objects.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_selected_extension_current_surface_available_now",
            "pass" if selected_extension_current_surface_available_now else "reject",
            "exact Trial-2 Ward / current algebra selected-extension current surface available now",
            sign_base.truth(selected_extension_current_surface_available_now),
            "Reject means the current selected-extension summaries do not actually materialize one J_sel^mu or current-vertex bundle independent of the blind scalar replay.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_selected_extension_ward_identity_surface_available_now",
            "pass" if selected_extension_ward_identity_surface_available_now else "reject",
            "exact Trial-2 Ward / current algebra selected-extension Ward-identity surface available now",
            sign_base.truth(selected_extension_ward_identity_surface_available_now),
            "Reject means no q_mu Gamma_sel^mu = 0 or equivalent Ward-identity surface is actually materialized on the current selected-extension pack.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_selected_extension_current_algebra_surface_available_now",
            "pass" if selected_extension_current_algebra_surface_available_now else "reject",
            "exact Trial-2 Ward / current algebra selected-extension current-algebra surface available now",
            sign_base.truth(selected_extension_current_algebra_surface_available_now),
            "Reject means no equal-time commutator, charge-generator, or current-algebra object is actually materialized on the current selected-extension pack.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_soft_limit_charge_normalization_collapse_theorem_available_now",
            "pass"
            if naive_soft_limit_charge_normalization_collapse_theorem_available_now
            else "reject",
            "exact Trial-2 Ward / current algebra soft-limit charge-normalization collapse theorem available now",
            sign_base.truth(
                naive_soft_limit_charge_normalization_collapse_theorem_available_now
            ),
            "The only current-pack soft readout still collapses to F(0)=1 and therefore to 1/(4 pi), not to one independent Ward/current-algebra alpha readout.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_independent_alpha_readout_available_now",
            "pass" if independent_ward_alpha_readout_available_now else "reject",
            "exact Trial-2 Ward / current algebra independent alpha readout available now",
            sign_base.truth(independent_ward_alpha_readout_available_now),
            "Reject means the route does not yet produce alpha from one selected-extension-native Ward/current-algebra surface; it only re-encounters charge normalization.",
        ),
        sign_base.row(
            "exact_trial2_ward_current_algebra_negative_closeout_available_now",
            "pass"
            if exact_trial2_ward_current_algebra_negative_closeout_available_now
            else "reject",
            "exact Trial-2 Ward / current algebra negative closeout available now",
            sign_base.truth(
                exact_trial2_ward_current_algebra_negative_closeout_available_now
            ),
            "The Ward/current-algebra route closes honestly because old Noether-current and source-no-go theorems survive, but no selected-extension-native Ward/current-algebra alpha surface is materialized.",
        ),
        sign_base.row(
            "updated_pack_trial2_conditional_reopen_refresh_required_now",
            "pass"
            if updated_pack_trial2_conditional_reopen_refresh_required_now
            else "reject",
            "updated-pack Trial-2 conditional-reopen refresh required now",
            sign_base.truth(
                updated_pack_trial2_conditional_reopen_refresh_required_now
            ),
            "Once Ward/current-algebra also closes negatively, the honest followup is to return to conditional reopen rather than inventing another current-pack branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": pack["selected_extension_label"],
        "soft_form_factor_zero": float(pack["soft_form_factor_zero"]),
        "soft_alpha_naive": float(pack["soft_alpha_naive"]),
        "alpha_target": float(pack["alpha_target"]),
        "soft_alpha_target_ratio": float(pack["soft_alpha_target_ratio"]),
        "soft_alpha_target_relative_mismatch": float(
            pack["soft_alpha_target_relative_mismatch"]
        ),
        "exact_trial2_ward_current_algebra_background_noether_current_available_now": (
            background_noether_current_available_now
        ),
        "exact_trial2_ward_current_algebra_same_field_source_no_go_available_now": (
            same_field_source_no_go_available_now
        ),
        "exact_trial2_ward_current_algebra_selected_extension_scalar_surface_available_now": (
            selected_extension_scalar_surface_available_now
        ),
        "selected_extension_current_hits": pack["selected_extension_current_hits"],
        "selected_extension_ward_hits": pack["selected_extension_ward_hits"],
        "selected_extension_current_algebra_hits": (
            pack["selected_extension_current_algebra_hits"]
        ),
        "exact_trial2_ward_current_algebra_selected_extension_current_surface_available_now": (
            selected_extension_current_surface_available_now
        ),
        "exact_trial2_ward_current_algebra_selected_extension_ward_identity_surface_available_now": (
            selected_extension_ward_identity_surface_available_now
        ),
        "exact_trial2_ward_current_algebra_selected_extension_current_algebra_surface_available_now": (
            selected_extension_current_algebra_surface_available_now
        ),
        "exact_trial2_ward_current_algebra_soft_limit_charge_normalization_collapse_theorem_available_now": (
            naive_soft_limit_charge_normalization_collapse_theorem_available_now
        ),
        "exact_trial2_ward_current_algebra_independent_alpha_readout_available_now": (
            independent_ward_alpha_readout_available_now
        ),
        "exact_trial2_ward_current_algebra_negative_closeout_available_now": (
            exact_trial2_ward_current_algebra_negative_closeout_available_now
        ),
        "updated_pack_trial2_conditional_reopen_refresh_required_now": (
            updated_pack_trial2_conditional_reopen_refresh_required_now
        ),
        "selected_primary_completion_lane": "trial2_ward_current_algebra_gate",
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "new_selected_extension_native_source_only",
        "selected_next_generation_route": "trial2_ward_current_algebra_gate",
        "recommended_next_route_or_none": "8.7.56.5563",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5561",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT
                    / "scripts"
                    / "quantum"
                    / "trial2_ward_current_algebra_route_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5563",
                "followup_route": "conditional_reopen_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_ward_current_algebra_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 Ward / current algebra audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から Ward/current-algebra audit を実行する。

if __name__ == "__main__":
    main()
