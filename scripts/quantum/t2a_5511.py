#!/usr/bin/env python3
"""Generate 8.7.56.5511-.5514 Trial-2 effective coupling / residue audit artifacts."""

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
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5507-5510",
        "updated_pack_trial2_spectral_distinguished_scale_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
RECOMP_IMPL = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5239-5242",
        "updated_pack_selected_extension_solver_recompute_implementation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
RECOMP_RERUN = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5247-5250",
        "updated_pack_selected_extension_solver_recompute_retained_q_rerun_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
DEFORM_IMPL = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5279-5282",
        "updated_pack_selected_extension_solver_side_deformation_front_runner_implementation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
DEFORM_RERUN = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5287-5290",
        "updated_pack_selected_extension_solver_side_deformation_numeric_rerun_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "68_trial2_numeric_alpha_vector_qball_effective_coupling_residue_audit.md"
)
RECOMP_BACKEND = (
    ROOT / "scripts" / "quantum" / "selected_extension_solver_recompute_backend.py"
)
DEFORM_BACKEND = (
    ROOT
    / "scripts"
    / "quantum"
    / "selected_extension_solver_side_deformation_backend.py"
)

STEP_TAG = "8.7.56.5511-5514"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "effective coupling / residue audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_effective_coupling_residue_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_spectral_distinguished_scale_target_free_negative_closeout_"
    "completed_residue_primary_source_materialization_reserve_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_effective_coupling_residue_current_pack_readout_negative_closeout_"
    "completed_conditional_hold_gate_next"
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
    """Return whether the note carries the expected residue-route claims."""
    patterns = (
        "effective coupling / residue route",
        "current vertex surface",
        "pole location / residue normalization",
        "blind-vector retained replay",
        "reopen-route inventory is exhausted",
    )
    return all(pattern in text for pattern in patterns)


# 関数: source text が recompute collapse を明示するか確認する。

def recompute_collapse_visible(text: str) -> bool:
    """Return whether recompute backend visibly collapses F_blind onto Z_eff."""
    return (
        "f_blind_recomp = dict(z_eff_recomp_transverse_scalar)" in text
        and 'alpha_blind_recomp_at_q_theory = float(\n        backend_pack["blind_target_keys"]["blind_alpha_at_q_theory"]' in text
    )


# 関数: source text が deformation preservation を明示するか確認する。

def deformation_preservation_visible(text: str) -> bool:
    """Return whether deformation backend visibly preserves the recompute surface."""
    return (
        'z_eff_deform_transverse_scalar = dict(\n        recompute_pack["Z_eff_recomp_transverse_scalar_pack"]' in text
        and 'f_blind_deform = dict(recompute_pack["F_blind_recomp_pack"])' in text
        and 'alpha_blind_deform_at_q_theory = float(\n        recompute_pack["alpha_blind_recomp_at_q_theory"]' in text
    )


# 関数: summary key set に current-vertex / pole-residue object があるかを確認する。

def has_independent_residue_keys(summary_keys: set[str]) -> tuple[bool, bool]:
    """Return current-vertex and pole/residue availability inferred from summary keys."""
    current_vertex = any("current_vertex" in key for key in summary_keys)
    pole_or_residue = any(
        ("pole" in key) or ("residue_normalization" in key) or ("jj_residue" in key)
        for key in summary_keys
    )
    return current_vertex, pole_or_residue


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the effective coupling / residue audit."""
    return {
        "recompute_pack": (
            "O_recomp_sel := {K_eff^(recomp)[Q_ret], Z_eff^(recomp,T)[Q_ret], "
            "F_blind^(recomp)[Q_ret], alpha_blind^(recomp)(q_theory)}"
        ),
        "collapse_rule": "F_blind^(recomp)[Q_ret] := Z_eff^(recomp,T)[Q_ret]",
        "deformation_rule": "O_deform_sel := O_recomp_sel on the retained current pack",
        "residue_requirement": (
            "alpha_res requires one independent selected-extension-native current-vertex / "
            "two-point pole / residue-normalization readout"
        ),
    }


# 関数: `.5511-.5514` を実行する。

def main() -> None:
    """Execute the Trial-2 effective coupling / residue audit."""
    for path in (
        PRIOR_GATE,
        RECOMP_IMPL,
        RECOMP_RERUN,
        DEFORM_IMPL,
        DEFORM_RERUN,
        AUDIT_NOTE,
        RECOMP_BACKEND,
        DEFORM_BACKEND,
    ):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    recomp_impl_summary = sign_base.read_json(RECOMP_IMPL)["summary"]
    recomp_rerun_summary = sign_base.read_json(RECOMP_RERUN)["summary"]
    deform_impl_summary = sign_base.read_json(DEFORM_IMPL)["summary"]
    deform_rerun_summary = sign_base.read_json(DEFORM_RERUN)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    recomp_source = sign_base.read_text(RECOMP_BACKEND)
    deform_source = sign_base.read_text(DEFORM_BACKEND)

    note_available = note_contains_audit(note_text)
    backend_surface_available_now = bool(
        prior_summary["trial2_effective_coupling_residue_primary_next_now"]
        and recomp_impl_summary[
            "exact_selected_extension_solver_recompute_materialized_output_pack_available_now"
        ]
        and deform_impl_summary[
            "exact_selected_extension_solver_side_deformation_materialized_output_pack_available_now"
        ]
    )
    recompute_collapse_now = bool(
        recompute_collapse_visible(recomp_source)
        and math.isclose(
            float(recomp_impl_summary["blind_alpha_recomp_at_q_theory"]),
            float(recomp_rerun_summary["blind_alpha_recomp_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(recomp_impl_summary["delta_alpha_sel_recomp_exact"]),
            float(recomp_rerun_summary["delta_alpha_sel_recomp_exact"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    deformation_preservation_now = bool(
        deformation_preservation_visible(deform_source)
        and math.isclose(
            float(recomp_rerun_summary["blind_alpha_recomp_at_q_theory"]),
            float(deform_rerun_summary["blind_alpha_deform_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(recomp_rerun_summary["delta_alpha_sel_recomp_exact"]),
            float(deform_rerun_summary["delta_alpha_sel_deform_exact"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(recomp_rerun_summary["relative_exact_residual_recomp"]),
            float(deform_rerun_summary["relative_exact_residual_deform"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )

    summary_keys = (
        set(recomp_impl_summary.keys())
        | set(recomp_rerun_summary.keys())
        | set(deform_impl_summary.keys())
        | set(deform_rerun_summary.keys())
    )
    independent_current_vertex_surface_available_now, independent_pole_residue_surface_available_now = (
        has_independent_residue_keys(summary_keys)
    )
    independent_alpha_readout_available_now = bool(
        backend_surface_available_now
        and not recompute_collapse_now
        and not deformation_preservation_now
        and independent_current_vertex_surface_available_now
        and independent_pole_residue_surface_available_now
    )
    effective_coupling_residue_lane_negative_closeout_available_now = bool(
        backend_surface_available_now
        and recompute_collapse_now
        and deformation_preservation_now
        and not independent_current_vertex_surface_available_now
        and not independent_pole_residue_surface_available_now
        and not independent_alpha_readout_available_now
    )
    updated_pack_trial2_reopen_route_inventory_exhausted_followup_required_now = bool(
        effective_coupling_residue_lane_negative_closeout_available_now
    )

    rows = [
        sign_base.row(
            "exact_trial2_effective_coupling_residue_audit_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 effective coupling / residue audit note available now",
            sign_base.truth(note_available),
            "The dedicated residue-route audit note exists and records the current-pack no-go logic.",
        ),
        sign_base.row(
            "exact_trial2_effective_coupling_residue_backend_surface_available_now",
            "pass" if backend_surface_available_now else "reject",
            "exact Trial-2 effective coupling / residue backend surface available now",
            sign_base.truth(backend_surface_available_now),
            "The selected-extension recompute/deformation helpers do materialize effective-kernel and transverse-scalar packs on the retained q window.",
        ),
        sign_base.row(
            "exact_trial2_effective_coupling_residue_recompute_collapse_theorem_available_now",
            "pass" if recompute_collapse_now else "reject",
            "exact Trial-2 effective coupling / residue recompute collapse theorem available now",
            sign_base.truth(recompute_collapse_now),
            "The recompute helper defines F_blind as a direct copy of Z_eff and carries alpha from the retained blind checkpoint.",
        ),
        sign_base.row(
            "exact_trial2_effective_coupling_residue_deformation_preservation_theorem_available_now",
            "pass" if deformation_preservation_now else "reject",
            "exact Trial-2 effective coupling / residue deformation preservation theorem available now",
            sign_base.truth(deformation_preservation_now),
            "The deformation helper preserves the recompute surface exactly instead of introducing one independent residue readout.",
        ),
        sign_base.row(
            "exact_trial2_effective_coupling_residue_independent_current_vertex_surface_available_now",
            "pass" if independent_current_vertex_surface_available_now else "reject",
            "exact Trial-2 effective coupling / residue independent current-vertex surface available now",
            sign_base.truth(independent_current_vertex_surface_available_now),
            "Reject means no selected-extension-native current-vertex surface is actually materialized in the public residue-route summaries.",
        ),
        sign_base.row(
            "exact_trial2_effective_coupling_residue_independent_pole_residue_surface_available_now",
            "pass" if independent_pole_residue_surface_available_now else "reject",
            "exact Trial-2 effective coupling / residue independent pole-residue surface available now",
            sign_base.truth(independent_pole_residue_surface_available_now),
            "Reject means no independent pole location or residue-normalization object is materialized beyond relative exact residual bookkeeping.",
        ),
        sign_base.row(
            "exact_trial2_effective_coupling_residue_new_alpha_readout_available_now",
            "pass" if independent_alpha_readout_available_now else "reject",
            "exact Trial-2 effective coupling / residue new alpha readout available now",
            sign_base.truth(independent_alpha_readout_available_now),
            "Reject means the current pack does not yet produce alpha from an independent residue normalization; it replays the blind surface.",
        ),
        sign_base.row(
            "exact_trial2_effective_coupling_residue_lane_negative_closeout_available_now",
            "pass"
            if effective_coupling_residue_lane_negative_closeout_available_now
            else "reject",
            "exact Trial-2 effective coupling / residue lane negative closeout available now",
            sign_base.truth(
                effective_coupling_residue_lane_negative_closeout_available_now
            ),
            "The residue route closes honestly under the current pack because no independent current-vertex/pole-residue readout is actually materialized.",
        ),
        sign_base.row(
            "updated_pack_trial2_reopen_route_inventory_exhausted_followup_required_now",
            "pass"
            if updated_pack_trial2_reopen_route_inventory_exhausted_followup_required_now
            else "reject",
            "updated-pack Trial-2 reopen-route inventory exhausted followup required now",
            sign_base.truth(
                updated_pack_trial2_reopen_route_inventory_exhausted_followup_required_now
            ),
            "Once residue also closes negatively, the honest followup is to restore conditional hold rather than inventing a new current-pack branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "blind_alpha_recomp_at_q_theory": float(
            recomp_rerun_summary["blind_alpha_recomp_at_q_theory"]
        ),
        "blind_alpha_deform_at_q_theory": float(
            deform_rerun_summary["blind_alpha_deform_at_q_theory"]
        ),
        "delta_alpha_sel_recomp_exact": float(
            recomp_rerun_summary["delta_alpha_sel_recomp_exact"]
        ),
        "delta_alpha_sel_deform_exact": float(
            deform_rerun_summary["delta_alpha_sel_deform_exact"]
        ),
        "relative_exact_residual_recomp": float(
            recomp_rerun_summary["relative_exact_residual_recomp"]
        ),
        "relative_exact_residual_deform": float(
            deform_rerun_summary["relative_exact_residual_deform"]
        ),
        "exact_trial2_effective_coupling_residue_audit_note_available_now": note_available,
        "exact_trial2_effective_coupling_residue_backend_surface_available_now": (
            backend_surface_available_now
        ),
        "exact_trial2_effective_coupling_residue_recompute_collapse_theorem_available_now": (
            recompute_collapse_now
        ),
        "exact_trial2_effective_coupling_residue_deformation_preservation_theorem_available_now": (
            deformation_preservation_now
        ),
        "exact_trial2_effective_coupling_residue_independent_current_vertex_surface_available_now": (
            independent_current_vertex_surface_available_now
        ),
        "exact_trial2_effective_coupling_residue_independent_pole_residue_surface_available_now": (
            independent_pole_residue_surface_available_now
        ),
        "exact_trial2_effective_coupling_residue_new_alpha_readout_available_now": (
            independent_alpha_readout_available_now
        ),
        "exact_trial2_effective_coupling_residue_lane_negative_closeout_available_now": (
            effective_coupling_residue_lane_negative_closeout_available_now
        ),
        "updated_pack_trial2_reopen_route_inventory_exhausted_followup_required_now": (
            updated_pack_trial2_reopen_route_inventory_exhausted_followup_required_now
        ),
        "selected_primary_completion_lane": "updated_pack_trial2_effective_coupling_residue_gate",
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "selected_extension_source_materialization",
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "trial2_effective_coupling_residue_gate"
        ),
        "recommended_next_route_or_none": "8.7.56.5515",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5513",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "recompute_implementation": sign_base.display_path(RECOMP_IMPL),
                "recompute_numeric_rerun": sign_base.display_path(RECOMP_RERUN),
                "deformation_implementation": sign_base.display_path(DEFORM_IMPL),
                "deformation_numeric_rerun": sign_base.display_path(DEFORM_RERUN),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
                "recompute_backend": sign_base.display_path(RECOMP_BACKEND),
                "deformation_backend": sign_base.display_path(DEFORM_BACKEND),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5515",
                "followup_route": "conditional_reopen_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_effective_coupling_residue_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 effective coupling / residue audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から effective coupling / residue audit を実行する。

if __name__ == "__main__":
    main()
