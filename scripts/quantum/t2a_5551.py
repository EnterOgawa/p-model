#!/usr/bin/env python3
"""Generate 8.7.56.5551-.5554 Trial-2 scattering / Thomson-limit audit artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_scattering_thomson_route_backend import (
    build_trial2_scattering_thomson_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5547-5550",
        "updated_pack_trial2_full_spectral_jost_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
RESIDUE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5511-5514",
        "updated_pack_trial2_effective_coupling_residue_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "73_trial2_numeric_alpha_vector_qball_scattering_thomson_audit.md"
)

STEP_TAG = "8.7.56.5551-5554"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "scattering amplitude / Thomson-limit audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_scattering_thomson_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_full_spectral_jost_negative_closeout_completed_scattering_thomson_"
    "primary_ward_current_algebra_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_scattering_thomson_target_free_readout_missing_"
    "ward_current_algebra_primary_gate_next"
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
    """Return whether the dedicated scattering audit note carries the required claims."""
    patterns = (
        "scattering amplitude / Thomson-limit route",
        "M_sel(omega, theta, pol)",
        "F(0)=1",
        "legacy Phase-3 sideband carry-over",
        "Ward identity / current algebra route",
    )
    return all(pattern in text for pattern in patterns)


# 関数: scattering audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the scattering / Thomson-limit audit."""
    return {
        "required_surface": (
            "S_scat := {M_sel(omega, theta, pol), M_T_sel(theta), sigma_T_sel}"
        ),
        "independent_readout": "alpha_scat := N_T[S_scat]",
        "soft_limit_collapse": (
            "alpha_soft,naive := F_blind(0)^2 / (4 pi) = 1 / (4 pi)"
        ),
        "negative_closeout_rule": (
            "scattering_negative_closeout iff no omega/theta/polarization/Thomson "
            "surface is materialized and the only near-target extra-q point is "
            "legacy Phase-3 sideband carry-over"
        ),
    }


# 関数: `.5551-.5554` を実行する。

def main() -> None:
    """Execute the Trial-2 scattering / Thomson-limit audit."""
    for path in (PRIOR_GATE, RESIDUE_AUDIT, AUDIT_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    residue_summary = sign_base.read_json(RESIDUE_AUDIT)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_scattering_thomson_pack()

    route_selected = bool(prior_summary["trial2_scattering_thomson_primary_next_now"])
    note_available = note_contains_audit(note_text)
    selected_extension_scalar_surface_available_now = bool(
        set(("zero", "q_theory_over_m0", "m0"))
        <= set(pack["retained_q_window"].keys())
        and isinstance(pack["best_extra_label_vs_alpha_target"], str)
    )
    independent_current_vertex_surface_available_now = bool(
        residue_summary[
            "exact_trial2_effective_coupling_residue_independent_current_vertex_surface_available_now"
        ]
    )
    low_energy_surface_available_now = bool(pack["low_energy_surface_available_now"])
    angular_surface_available_now = bool(pack["angular_surface_available_now"])
    polarization_surface_available_now = bool(
        pack["polarization_surface_available_now"]
    )
    thomson_surface_available_now = bool(pack["thomson_surface_available_now"])
    naive_soft_limit_charge_normalization_collapse_theorem_available_now = bool(
        pack["naive_soft_limit_charge_normalization_collapse_now"]
    )
    legacy_phase3_sideband_target_proximity_only_now = bool(
        pack["legacy_phase3_sideband_target_proximity_only_now"]
    )
    independent_scattering_thomson_alpha_readout_available_now = bool(
        pack["independent_scattering_surface_available_now"]
        and independent_current_vertex_surface_available_now
    )
    exact_trial2_scattering_thomson_negative_closeout_available_now = bool(
        route_selected
        and note_available
        and selected_extension_scalar_surface_available_now
        and not low_energy_surface_available_now
        and not angular_surface_available_now
        and not polarization_surface_available_now
        and not thomson_surface_available_now
        and naive_soft_limit_charge_normalization_collapse_theorem_available_now
        and legacy_phase3_sideband_target_proximity_only_now
        and not independent_scattering_thomson_alpha_readout_available_now
    )
    updated_pack_trial2_ward_current_algebra_followup_required_now = bool(
        exact_trial2_scattering_thomson_negative_closeout_available_now
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_scattering_thomson_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 scattering / Thomson route selected now",
            sign_base.truth(route_selected),
            "The route is worth auditing only after Jost has closed negatively and scattering / Thomson has been promoted to primary.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 scattering / Thomson audit note available now",
            sign_base.truth(note_available),
            "The dedicated scattering / Thomson audit note exists and records the current-pack no-go logic.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_selected_extension_scalar_surface_available_now",
            "pass" if selected_extension_scalar_surface_available_now else "reject",
            "exact Trial-2 scattering / Thomson selected-extension scalar surface available now",
            sign_base.truth(selected_extension_scalar_surface_available_now),
            "The current pack does materialize retained-q and helper-backed extra-q scalar checkpoints; the question is whether they define one independent scattering surface.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_independent_current_vertex_surface_available_now",
            "pass" if independent_current_vertex_surface_available_now else "reject",
            "exact Trial-2 scattering / Thomson independent current-vertex surface available now",
            sign_base.truth(independent_current_vertex_surface_available_now),
            "Reject means the current pack still lacks one selected-extension-native current-vertex surface independent of the blind replay.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_low_energy_surface_available_now",
            "pass" if low_energy_surface_available_now else "reject",
            "exact Trial-2 scattering / Thomson low-energy omega surface available now",
            sign_base.truth(low_energy_surface_available_now),
            "Reject means no omega-resolved scattering surface is materialized in the current selected-extension pack.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_angular_surface_available_now",
            "pass" if angular_surface_available_now else "reject",
            "exact Trial-2 scattering / Thomson angular surface available now",
            sign_base.truth(angular_surface_available_now),
            "Reject means no theta/angle-resolved scattering object is materialized.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_polarization_surface_available_now",
            "pass" if polarization_surface_available_now else "reject",
            "exact Trial-2 scattering / Thomson polarization surface available now",
            sign_base.truth(polarization_surface_available_now),
            "Reject means no polarization/helicity-resolved scattering object is materialized.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_cross_section_surface_available_now",
            "pass" if thomson_surface_available_now else "reject",
            "exact Trial-2 scattering / Thomson cross-section surface available now",
            sign_base.truth(thomson_surface_available_now),
            "Reject means no Thomson/cross-section normalization object is materialized independently of blind form-factor packs.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_soft_limit_charge_normalization_collapse_theorem_available_now",
            "pass"
            if naive_soft_limit_charge_normalization_collapse_theorem_available_now
            else "reject",
            "exact Trial-2 scattering / Thomson soft-limit charge-normalization collapse theorem available now",
            sign_base.truth(
                naive_soft_limit_charge_normalization_collapse_theorem_available_now
            ),
            "The naive soft readout collapses to F(0)=1 and therefore to 1/(4 pi), not to the physical alpha target.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_legacy_phase3_sideband_target_proximity_only_now",
            "pass" if legacy_phase3_sideband_target_proximity_only_now else "reject",
            "exact Trial-2 scattering / Thomson legacy Phase-3 sideband target proximity only now",
            sign_base.truth(legacy_phase3_sideband_target_proximity_only_now),
            "The only near-target extra-q point remains best_global_signed_q from legacy Phase-3 carry-over, not one selected-extension-native Thomson surface.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_independent_alpha_readout_available_now",
            "pass" if independent_scattering_thomson_alpha_readout_available_now else "reject",
            "exact Trial-2 scattering / Thomson independent alpha readout available now",
            sign_base.truth(independent_scattering_thomson_alpha_readout_available_now),
            "Reject means no omega/theta/polarization-normalized scattering readout exists beyond the blind scalar packs.",
        ),
        sign_base.row(
            "exact_trial2_scattering_thomson_negative_closeout_available_now",
            "pass"
            if exact_trial2_scattering_thomson_negative_closeout_available_now
            else "reject",
            "exact Trial-2 scattering / Thomson negative closeout available now",
            sign_base.truth(
                exact_trial2_scattering_thomson_negative_closeout_available_now
            ),
            "The scattering / Thomson route closes honestly because current pack materializes neither one independent scattering surface nor one Thomson-limit alpha readout.",
        ),
        sign_base.row(
            "updated_pack_trial2_ward_current_algebra_followup_required_now",
            "pass"
            if updated_pack_trial2_ward_current_algebra_followup_required_now
            else "reject",
            "updated-pack Trial-2 Ward / current algebra followup required now",
            sign_base.truth(
                updated_pack_trial2_ward_current_algebra_followup_required_now
            ),
            "Once scattering closes negatively, the honest next primary route is Ward / current algebra.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_extension_label": pack["selected_extension_label"],
        "q_theory_over_m0": float(pack["q_theory_over_m0"]),
        "q_theory_form_factor": float(pack["q_theory_form_factor"]),
        "q_theory_alpha": float(pack["q_theory_alpha"]),
        "soft_form_factor_zero": float(pack["soft_form_factor_zero"]),
        "soft_alpha_naive": float(pack["soft_alpha_naive"]),
        "alpha_target": float(pack["alpha_target"]),
        "soft_alpha_target_ratio": float(pack["soft_alpha_target_ratio"]),
        "soft_alpha_target_relative_mismatch": float(
            pack["soft_alpha_target_relative_mismatch"]
        ),
        "best_extra_label_vs_alpha_target": pack["best_extra_label_vs_alpha_target"],
        "best_extra_q_over_m0": float(pack["best_extra_q_over_m0"]),
        "best_extra_alpha": float(pack["best_extra_alpha"]),
        "best_extra_alpha_target_residual": float(
            pack["best_extra_alpha_target_residual"]
        ),
        "best_extra_legacy_phase3_sideband": bool(
            pack["best_extra_legacy_phase3_sideband"]
        ),
        "low_energy_surface_hits": pack["low_energy_surface_hits"],
        "angular_surface_hits": pack["angular_surface_hits"],
        "polarization_surface_hits": pack["polarization_surface_hits"],
        "thomson_surface_hits": pack["thomson_surface_hits"],
        "exact_trial2_scattering_thomson_selected_extension_scalar_surface_available_now": (
            selected_extension_scalar_surface_available_now
        ),
        "exact_trial2_scattering_thomson_independent_current_vertex_surface_available_now": (
            independent_current_vertex_surface_available_now
        ),
        "exact_trial2_scattering_thomson_low_energy_surface_available_now": (
            low_energy_surface_available_now
        ),
        "exact_trial2_scattering_thomson_angular_surface_available_now": (
            angular_surface_available_now
        ),
        "exact_trial2_scattering_thomson_polarization_surface_available_now": (
            polarization_surface_available_now
        ),
        "exact_trial2_scattering_thomson_cross_section_surface_available_now": (
            thomson_surface_available_now
        ),
        "exact_trial2_scattering_thomson_soft_limit_charge_normalization_collapse_theorem_available_now": (
            naive_soft_limit_charge_normalization_collapse_theorem_available_now
        ),
        "exact_trial2_scattering_thomson_legacy_phase3_sideband_target_proximity_only_now": (
            legacy_phase3_sideband_target_proximity_only_now
        ),
        "exact_trial2_scattering_thomson_independent_alpha_readout_available_now": (
            independent_scattering_thomson_alpha_readout_available_now
        ),
        "exact_trial2_scattering_thomson_negative_closeout_available_now": (
            exact_trial2_scattering_thomson_negative_closeout_available_now
        ),
        "updated_pack_trial2_ward_current_algebra_followup_required_now": (
            updated_pack_trial2_ward_current_algebra_followup_required_now
        ),
        "selected_primary_completion_lane": "trial2_scattering_thomson_gate",
        "selected_secondary_completion_lane": "trial2_ward_current_algebra",
        "selected_reserve_completion_lane": "conditional_reopen_only",
        "selected_next_generation_route": "trial2_scattering_thomson_gate",
        "recommended_next_route_or_none": "8.7.56.5555",
        "selected_followup_route": "trial2_ward_current_algebra",
        "selected_followup_route_or_none": "8.7.56.5559",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5553",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "residue_audit": sign_base.display_path(RESIDUE_AUDIT),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "backend_helper": sign_base.display_path(
                    ROOT
                    / "scripts"
                    / "quantum"
                    / "trial2_scattering_thomson_route_backend.py"
                ),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5555",
                "followup_route": "8.7.56.5559",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_scattering_thomson_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 scattering / Thomson audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から scattering audit を実行する。

if __name__ == "__main__":
    main()
