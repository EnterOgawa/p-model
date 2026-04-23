#!/usr/bin/env python3
"""Generate 8.7.56.1627-.1630 energy-density form-factor audit artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths

import scripts.quantum.t2a_1479 as exact_reinjection_tools
import scripts.quantum.t2a_1599 as exact_profile_tools


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

ENERGY_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_energy_density_formfactor_20260328.md"
)
ENERGY_DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1623_1626_energy_density_audit_declaration_gate_metrics.json"
)
CASEB_RESTORE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1619_1622_eff_metric_v2_sub_restore_declaration_gate_metrics.json"
)
ANCHOR_EVAL = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
)
PHASE1_EVAL = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_numeric_evaluation_metrics.json"
)

STEP_TAG = "8.7.56.1627-1630"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor energy-density form factor audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "energy_density_ff_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_energy_density_hamiltonian_core_derived_"
    "formfactor_audit_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_energy_density_form_factor_no_scalar_rescue_"
    "case_classification_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_alpha_case_classification"
)
NEXT_ROUTE = "8.7.56.1631"
NEXT_SYNC_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_disposition_sync_closeout"
)
NEXT_SYNC_ROUTE = "8.7.56.1635"

TARGET_ALPHA = 1.0 / 137.035999084
SCALAR_F = 0.2998913524347805
SCALAR_ALPHA = 0.00715678583937324
VECTOR_F = -0.083735013520183
VECTOR_ALPHA = 0.0005579616187042394
VACUUM_VEV_SQUARED = 1.0


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を構成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を構成する。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON/CSV 成果物を書き出す。

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

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: alpha の target 相対残差を返す。

def alpha_residual_rel(alpha_value: float) -> float:
    """Return one target-relative alpha residual."""
    return float(abs(float(alpha_value) - TARGET_ALPHA) / TARGET_ALPHA)


# 関数: retained branch の density bundle を再構成する。

def build_density_bundle() -> dict:
    """Reconstruct the retained exact branch and all audit density surfaces."""
    anchor_summary = read_json(ANCHOR_EVAL)["summary"]
    phase1_summary = read_json(PHASE1_EVAL)["summary"]
    phase1_row = anchor_summary["phase1_equivalent_row"]
    beta = float(anchor_summary["beta_1_scalar"])
    amp0 = float(phase1_summary["phase1_best_alpha_candidate"]["amp0"])
    amp_l = float(phase1_row["amp_l"])

    pivot = exact_reinjection_tools.load_module(
        exact_reinjection_tools.PIVOT_BRANCH,
        "pivot_branch_energy_density_ff_reuse",
    )
    profile = exact_profile_tools.solve_exact_profile_with_arrays(pivot, beta, amp0, amp_l)
    radius = np.asarray(profile["radius"], dtype=float)
    f0_values = np.asarray(profile["f0"], dtype=float)
    f_l_values = np.asarray(profile["fL"], dtype=float)
    f0_prime = np.gradient(f0_values, radius)
    f_l_prime = np.gradient(f_l_values, radius)
    q_theory = float(profile["q_theory_over_m0"])
    omega_sq = float(beta * beta)
    radial_mass_sq = float(pivot.RADIAL_MASS_SQUARED)
    radial_floor = np.maximum(radius * radius, 1.0e-12)

    electric_like_density = (f0_prime * f0_prime) + (omega_sq * f_l_values * f_l_values)
    radial_mass_density = radial_mass_sq * f0_values * f0_values
    hamiltonian_core_density = electric_like_density + radial_mass_density
    note_temporal_density = omega_sq * (f0_values * f0_values + f_l_values * f_l_values)
    note_gradient_density = (
        (f0_prime * f0_prime)
        + (f_l_prime * f_l_prime)
        + (2.0 * f_l_values * f_l_values / radial_floor)
    )
    scalar_proxy_density = f0_values * f0_values
    signed_density = scalar_proxy_density - (f_l_values * f_l_values)
    rho_squared_family_proxy = scalar_proxy_density + (f_l_values * f_l_values)
    mh_family_proxy_density = 0.25 * (
        (rho_squared_family_proxy - VACUUM_VEV_SQUARED) ** 2
        - (0.0 - VACUUM_VEV_SQUARED) ** 2
    )

    return {
        "radius": radius,
        "f0_values": f0_values,
        "f_l_values": f_l_values,
        "f0_prime": f0_prime,
        "f_l_prime": f_l_prime,
        "q_theory_over_m0": q_theory,
        "beta": beta,
        "amp0": amp0,
        "amp_l": amp_l,
        "omega_sq": omega_sq,
        "radial_mass_sq": radial_mass_sq,
        "hamiltonian_core_density": hamiltonian_core_density,
        "radial_mass_density": radial_mass_density,
        "electric_like_density": electric_like_density,
        "note_temporal_density": note_temporal_density,
        "note_gradient_density": note_gradient_density,
        "scalar_proxy_density": scalar_proxy_density,
        "signed_density": signed_density,
        "mh_family_proxy_density": mh_family_proxy_density,
    }


# 関数: density surface ごとの form-factor summary を返す。

def summarize_surface(radius: np.ndarray, density: np.ndarray, q_ratio: float) -> dict:
    """Evaluate one spherical form factor surface and derived alpha metrics."""
    form_value, norm_value = exact_profile_tools.form_factor(radius, density, q_ratio)
    alpha_value = float((form_value * form_value) / (4.0 * math.pi))
    return {
        "F_at_q_theory": float(form_value),
        "alpha_at_q_theory": alpha_value,
        "alpha_residual_rel": alpha_residual_rel(alpha_value),
        "norm": float(norm_value),
    }


# 関数: current-pack formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return the exact energy-density form-factor formulas used here."""
    return {
        "retained_ansatz": (
            "P_mu^Qball = (f_0(r) e^{i omega t}, f_L(r) r_hat_i e^{i omega t})"
        ),
        "radial_field_strength": (
            "F_0r^(P) = partial_0 P_r - partial_r P_0 = i omega f_L - f_0'"
        ),
        "longitudinal_curl_rule": "F_ij^(P) = 0 for the pure radial longitudinal ansatz f_L(r) r_hat_i",
        "official_energy_density": (
            "epsilon_H,core(r) = |F_0r^(P)|^2 + m_0^2 f_0(r)^2 "
            "= f_0'(r)^2 + omega^2 f_L(r)^2 + m_0^2 f_0(r)^2"
        ),
        "radial_mass_component": "epsilon_mass(r) = m_0^2 f_0(r)^2",
        "electric_like_component": "epsilon_el(r) = f_0'(r)^2 + omega^2 f_L(r)^2",
        "energy_form_factor": (
            "F_E(q) = int epsilon(r) sinc(q r) r^2 dr / int epsilon(r) r^2 dr"
        ),
        "energy_alpha_rule": "alpha_E(q) = F_E(q)^2 / (4 pi)",
        "note_temporal_surface": "epsilon_note,temp = omega^2 (f_0^2 + f_L^2)",
        "note_gradient_surface": "epsilon_note,grad = f_0'^2 + f_L'^2 + 2 f_L^2 / r^2",
        "family_proxy_surface": (
            "epsilon_MH,proxy = (1/4)[(rho^2-v^2)^2 - v^4], rho^2 = f_0^2 + f_L^2"
        ),
        "exact_gap": (
            "The branch-local constitutive map rho[f_0,f_L] is still unavailable, so "
            "the full exact nonlinear energy density is not yet fixed on the restored "
            "vector branch."
        ),
    }


# 関数: `.1627-.1630` を実行する。

def main() -> None:
    """Execute the energy-density form-factor audit branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART1,
        PART5,
        ENERGY_NOTE,
        ENERGY_DERIV_GATE,
        CASEB_RESTORE_GATE,
        ANCHOR_EVAL,
        PHASE1_EVAL,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    energy_note_text = read_text(ENERGY_NOTE)

    energy_deriv_summary = read_json(ENERGY_DERIV_GATE)["summary"]
    caseb_restore_summary = read_json(CASEB_RESTORE_GATE)["summary"]
    bundle = build_density_bundle()

    core_surface = summarize_surface(
        bundle["radius"],
        bundle["hamiltonian_core_density"],
        bundle["q_theory_over_m0"],
    )
    radial_mass_surface = summarize_surface(
        bundle["radius"],
        bundle["radial_mass_density"],
        bundle["q_theory_over_m0"],
    )
    electric_like_surface = summarize_surface(
        bundle["radius"],
        bundle["electric_like_density"],
        bundle["q_theory_over_m0"],
    )
    note_temporal_surface = summarize_surface(
        bundle["radius"],
        bundle["note_temporal_density"],
        bundle["q_theory_over_m0"],
    )
    note_gradient_surface = summarize_surface(
        bundle["radius"],
        bundle["note_gradient_density"],
        bundle["q_theory_over_m0"],
    )
    mh_family_proxy_surface = summarize_surface(
        bundle["radius"],
        bundle["mh_family_proxy_density"],
        bundle["q_theory_over_m0"],
    )

    exact_energy_core_form_factor_available = True
    energy_core_tracks_vector_no_go_scale = bool(
        abs(core_surface["alpha_at_q_theory"] - VECTOR_ALPHA) / VECTOR_ALPHA <= 0.05
    )
    energy_core_supports_scalar_candidate = bool(
        core_surface["alpha_residual_rel"] <= 0.05
    )
    energy_core_exact_foundation_supported = bool(
        core_surface["alpha_residual_rel"] <= 0.01
    )
    electric_like_component_subleading = bool(
        energy_deriv_summary["electric_like_term_fraction"] < 0.01
    )
    electric_like_improves_but_is_not_official = bool(
        electric_like_surface["alpha_residual_rel"] < core_surface["alpha_residual_rel"]
    )
    note_temporal_surface_supports_scalar_candidate = bool(
        note_temporal_surface["alpha_residual_rel"] <= 0.05
    )
    note_gradient_surface_supports_scalar_candidate = bool(
        note_gradient_surface["alpha_residual_rel"] <= 0.05
    )
    heuristic_mh_family_proxy_available = True
    heuristic_mh_family_proxy_supports_scalar_candidate = bool(
        mh_family_proxy_surface["alpha_residual_rel"] <= 0.05
    )
    branch_local_full_energy_density_available = False
    energy_density_alpha_case_classification_admissible_now = True
    physical_reject_required = False

    core_vs_vector_alpha_rel_gap = float(
        abs(core_surface["alpha_at_q_theory"] - VECTOR_ALPHA) / VECTOR_ALPHA
    )
    core_vs_scalar_alpha_rel_gap = float(
        abs(core_surface["alpha_at_q_theory"] - SCALAR_ALPHA) / SCALAR_ALPHA
    )
    core_vs_vector_form_factor_gap = float(abs(core_surface["F_at_q_theory"] - VECTOR_F))
    core_vs_scalar_form_factor_gap = float(abs(core_surface["F_at_q_theory"] - SCALAR_F))

    rows = [
        row(
            "exact_energy_core_form_factor_available",
            "pass" if exact_energy_core_form_factor_available else "reject",
            "exact Hamiltonian-core form factor available",
            truth(exact_energy_core_form_factor_available),
            "The exact current-pack Hamiltonian core is already derived, so the corresponding spherical form factor can now be evaluated without adding new structure.",
        ),
        row(
            "energy_core_alpha_at_q_theory",
            "watch",
            "official energy-core alpha at q_theory",
            core_surface["alpha_at_q_theory"],
            "This is the official energy-density branch read because epsilon_H,core is the only exact current-pack positive-density observable now fixed on the restored vector branch.",
        ),
        row(
            "energy_core_residual_rel_vs_target",
            "reject" if not energy_core_supports_scalar_candidate else "pass",
            "official energy-core alpha relative residual vs target",
            core_surface["alpha_residual_rel"],
            "The official core observable remains far from the target, so the energy-density branch does not yet rescue the scalar candidate under the current pack.",
        ),
        row(
            "energy_core_tracks_vector_no_go_scale",
            "pass" if energy_core_tracks_vector_no_go_scale else "watch",
            "official energy-core alpha tracks vector no-go scale",
            core_vs_vector_alpha_rel_gap,
            "The exact Hamiltonian-core alpha lands much closer to the retained vector no-go scale than to the retained scalar strong candidate.",
        ),
        row(
            "energy_core_vs_scalar_alpha_rel_gap",
            "reject",
            "official energy-core alpha relative gap vs retained scalar alpha",
            core_vs_scalar_alpha_rel_gap,
            "This quantifies how far the official energy-core observable remains from the retained scalar strong candidate.",
        ),
        row(
            "radial_mass_component_alpha_at_q_theory",
            "watch",
            "radial mass component alpha at q_theory",
            radial_mass_surface["alpha_at_q_theory"],
            "Because the radial mass component is a constant multiple of f_0^2, its normalized form factor shows what the dominant positive piece contributes by itself on the restored vector branch.",
        ),
        row(
            "electric_like_component_subleading",
            "pass" if electric_like_component_subleading else "watch",
            "electric-like component is norm-subleading",
            energy_deriv_summary["electric_like_term_fraction"],
            "The electric-like correction is exact but subleading in the norm budget, so it cannot be promoted over the official Hamiltonian core read.",
        ),
        row(
            "electric_like_component_improves_but_is_not_official",
            "watch" if electric_like_improves_but_is_not_official else "reject",
            "electric-like component improves residual but is not official surface",
            electric_like_surface["alpha_residual_rel"],
            "The electric-like piece alone is numerically closer to target than the full core, but it is not an honest standalone observable because the exact official surface is the full positive Hamiltonian core.",
        ),
        row(
            "note_temporal_surface_supports_scalar_candidate",
            "pass" if note_temporal_surface_supports_scalar_candidate else "reject",
            "note temporal evidence surface supports scalar candidate",
            note_temporal_surface["alpha_residual_rel"],
            "Even as evidence only, the note's temporal surface stays near the vector no-go scale and does not rescue the scalar candidate.",
        ),
        row(
            "note_gradient_surface_supports_scalar_candidate",
            "pass" if note_gradient_surface_supports_scalar_candidate else "reject",
            "note gradient evidence surface supports scalar candidate",
            note_gradient_surface["alpha_residual_rel"],
            "The note's gradient surface improves over the official core but still misses the target by a large margin and is not exact under the current pack.",
        ),
        row(
            "heuristic_mh_family_proxy_supports_scalar_candidate",
            "pass" if heuristic_mh_family_proxy_supports_scalar_candidate else "reject",
            "heuristic Mexican-hat family proxy supports scalar candidate",
            mh_family_proxy_surface["alpha_residual_rel"],
            "Even the evidence-only vacuum-subtracted family proxy does not show a hidden scalar rescue on the retained vector branch.",
        ),
        row(
            "branch_local_full_energy_density_available",
            "pass" if branch_local_full_energy_density_available else "reject",
            "branch-local full exact energy density available",
            truth(branch_local_full_energy_density_available),
            "The exact branch-local constitutive map rho[f_0,f_L] is still unavailable, so the full nonlinear energy density cannot yet be audited as an official exact surface.",
        ),
        row(
            "energy_density_alpha_case_classification_admissible_now",
            "pass" if energy_density_alpha_case_classification_admissible_now else "reject",
            "energy-density alpha case classification admissible now",
            truth(energy_density_alpha_case_classification_admissible_now),
            "The exact official surface has now been evaluated honestly, so the next branch can classify the energy-density read without adding another derivation layer.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "energy_note": display_path(ENERGY_NOTE),
            "energy_derivation_gate": display_path(ENERGY_DERIV_GATE),
            "caseb_restore_gate": display_path(CASEB_RESTORE_GATE),
            "anchor_eval": display_path(ANCHOR_EVAL),
            "phase1_eval": display_path(PHASE1_EVAL),
        },
        "constants": {
            "q_theory_over_m0": bundle["q_theory_over_m0"],
            "beta_1_scalar": bundle["beta"],
            "amp0_phase1_equivalent": bundle["amp0"],
            "amp_l_phase1_equivalent": bundle["amp_l"],
            "radial_mass_squared": bundle["radial_mass_sq"],
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "official_surface_name": "epsilon_H_core",
        "official_F_E_at_q_theory": core_surface["F_at_q_theory"],
        "official_alpha_E_at_q_theory": core_surface["alpha_at_q_theory"],
        "official_alpha_E_residual_rel": core_surface["alpha_residual_rel"],
        "energy_core_tracks_vector_no_go_scale": energy_core_tracks_vector_no_go_scale,
        "energy_core_supports_scalar_candidate": energy_core_supports_scalar_candidate,
        "energy_core_exact_foundation_supported": energy_core_exact_foundation_supported,
        "core_vs_vector_alpha_rel_gap": core_vs_vector_alpha_rel_gap,
        "core_vs_scalar_alpha_rel_gap": core_vs_scalar_alpha_rel_gap,
        "core_vs_vector_form_factor_gap": core_vs_vector_form_factor_gap,
        "core_vs_scalar_form_factor_gap": core_vs_scalar_form_factor_gap,
        "radial_mass_component_alpha_at_q_theory": radial_mass_surface["alpha_at_q_theory"],
        "electric_like_component_alpha_at_q_theory": electric_like_surface["alpha_at_q_theory"],
        "note_temporal_alpha_at_q_theory": note_temporal_surface["alpha_at_q_theory"],
        "note_gradient_alpha_at_q_theory": note_gradient_surface["alpha_at_q_theory"],
        "mh_family_proxy_alpha_at_q_theory": mh_family_proxy_surface["alpha_at_q_theory"],
        "electric_like_component_subleading": electric_like_component_subleading,
        "electric_like_improves_but_is_not_official": electric_like_improves_but_is_not_official,
        "note_temporal_surface_supports_scalar_candidate": note_temporal_surface_supports_scalar_candidate,
        "note_gradient_surface_supports_scalar_candidate": note_gradient_surface_supports_scalar_candidate,
        "heuristic_mh_family_proxy_available": heuristic_mh_family_proxy_available,
        "heuristic_mh_family_proxy_supports_scalar_candidate": heuristic_mh_family_proxy_supports_scalar_candidate,
        "branch_local_full_energy_density_available": branch_local_full_energy_density_available,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_sync_route": NEXT_SYNC_ROUTE_NAME,
        "recommended_followup_sync_route_or_none": NEXT_SYNC_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "energy_note_density_heading": hit(energy_note_text, "Energy density:"),
            "energy_note_temporal_formula": hit(
                energy_note_text, "\\varepsilon_{\\rm temporal}"
            ),
            "energy_note_gradient_formula": hit(
                energy_note_text, "\\varepsilon_{\\rm gradient}"
            ),
            "energy_derivation_exact_core": {
                "pattern": "exact_hamiltonian_core",
                "line": 1,
                "text": read_json(ENERGY_DERIV_GATE)["evidence"]["formulas"]["exact_hamiltonian_core"],
            },
            "status_current_step": hit(status_text, "8.7.56.1627"),
            "roadmap_current_branch": hit(roadmap_text, "energy-density form factor audit"),
            "current_problem_energy_branch": hit(
                current_problem_text, "energy-density form factor audit"
            ),
            "current_status_energy_branch": hit(
                current_status_text, "energy-density form factor audit"
            ),
            "unified_roadmap_energy_branch": hit(
                unified_roadmap_text, "`.1627-.1630` は **energy-density form factor audit**"
            ),
            "part5_energy_branch": hit(part5_text, "**energy-density form factor audit**"),
            "part1_pref_equals_infty": hit(part1_text, "P_{\\mathrm{ref}}\\equiv P_{\\infty}"),
        },
        "official_surface": core_surface,
        "radial_mass_surface": radial_mass_surface,
        "electric_like_surface": electric_like_surface,
        "note_temporal_surface": note_temporal_surface,
        "note_gradient_surface": note_gradient_surface,
        "mh_family_proxy_surface": mh_family_proxy_surface,
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "prior_caseb_restore_alpha": float(caseb_restore_summary["full_sub_alpha_at_q_theory"]),
            "prior_caseb_restore_residual_rel": float(caseb_restore_summary["full_sub_residual_rel"]),
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1627",
                f"{STEP_NAME} inventory",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "audit": write_artifact(
            "audit",
            payload(
                "8.7.56.1628",
                f"{STEP_NAME} audit",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1629",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload(
                "8.7.56.1630",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(json.dumps({"step": STEP_TAG, "stem": STEM, "artifacts": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
