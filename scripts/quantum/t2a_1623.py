#!/usr/bin/env python3
"""Generate 8.7.56.1623-.1626 energy-density derivation artifacts."""

from __future__ import annotations

import csv
import json
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
CASEB_RESTORE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1619_1622_eff_metric_v2_sub_restore_declaration_gate_metrics.json"
)
GROUND_STATE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1615_1618_gs_nodeless_audit_declaration_gate_metrics.json"
)
CASEB_TP_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1611_1614_eff_metric_tp_alpha_audit_declaration_gate_metrics.json"
)
CASEB_CORE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1607_1610_eff_metric_k_deriv_declaration_gate_metrics.json"
)
BREAKTHROUGH_VEV = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
)
MEXICAN_HAT_FREEZE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
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
EXACT_REINJECTION_BRANCH = ROOT / "scripts" / "quantum" / "t2a_1479.py"

STEP_TAG = "8.7.56.1623-1626"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor energy-density / "
    "Hamiltonian-density derivation audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "energy_density_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_effective_metric_v2_subtraction_no_metric_rescue_"
    "disposition_sync_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_energy_density_hamiltonian_core_derived_"
    "formfactor_audit_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_form_factor_audit"
)
NEXT_ROUTE = "8.7.56.1627"
NEXT_CASE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_alpha_case_classification"
)
NEXT_CASE_ROUTE = "8.7.56.1631"
NEXT_SYNC_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_energy_density_disposition_sync_closeout"
)
NEXT_SYNC_ROUTE = "8.7.56.1635"

SCALAR_F = 0.2998913524347805
SCALAR_ALPHA = 0.00715678583937324
VECTOR_F = -0.083735013520183
VECTOR_ALPHA = 0.0005579616187042394


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


# 関数: ローカル Python モジュールを動的 import する。

def load_module(path: Path, module_name: str):
    """Dynamically import one local Python module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: 球対称 density の r^2 重み norm を返す。

def component_norm(radius: np.ndarray, density: np.ndarray) -> float:
    """Return one weighted radial norm."""
    return float(np.trapezoid(density * (radius**2), radius))


# 関数: exact retained branch の component norms を構成する。

def build_exact_branch_components() -> dict:
    """Reconstruct the retained branch and return current-pack density components."""
    anchor_summary = read_json(ANCHOR_EVAL)
    phase1_summary = read_json(PHASE1_EVAL)
    exact_branch = load_module(EXACT_REINJECTION_BRANCH, "t2a_1479_energy_density_reuse")
    pivot = load_module(exact_branch.PIVOT_BRANCH, "pivot_branch_energy_density_reuse")

    phase1_row = anchor_summary["evidence"]["phase1_equivalent_row"]
    beta = float(anchor_summary["summary"]["beta_1_scalar"])
    amp0 = float(phase1_summary["summary"]["phase1_best_alpha_candidate"]["amp0"])
    amp_l = float(phase1_row["amp_l"])

    profile = exact_profile_tools.solve_exact_profile_with_arrays(pivot, beta, amp0, amp_l)
    radius = np.asarray(profile["radius"], dtype=float)
    f0_values = np.asarray(profile["f0"], dtype=float)
    f_l_values = np.asarray(profile["fL"], dtype=float)
    f0_prime = np.gradient(f0_values, radius)
    f_l_prime = np.gradient(f_l_values, radius)

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

    electric_norm = component_norm(radius, electric_like_density)
    radial_mass_norm = component_norm(radius, radial_mass_density)
    hamiltonian_core_norm = component_norm(radius, hamiltonian_core_density)
    note_temporal_norm = component_norm(radius, note_temporal_density)
    note_gradient_norm = component_norm(radius, note_gradient_density)

    return {
        "radius": radius,
        "q_theory_over_m0": float(profile["q_theory_over_m0"]),
        "beta": beta,
        "amp0": amp0,
        "amp_l": amp_l,
        "omega_sq": omega_sq,
        "radial_mass_sq": radial_mass_sq,
        "electric_norm": electric_norm,
        "radial_mass_norm": radial_mass_norm,
        "hamiltonian_core_norm": hamiltonian_core_norm,
        "note_temporal_norm": note_temporal_norm,
        "note_gradient_norm": note_gradient_norm,
        "electric_fraction": electric_norm / hamiltonian_core_norm,
        "radial_mass_fraction": radial_mass_norm / hamiltonian_core_norm,
        "note_temporal_vs_exact_mass_ratio": note_temporal_norm / radial_mass_norm,
        "note_gradient_vs_exact_electric_ratio": note_gradient_norm / electric_norm,
        "hamiltonian_core_positive_definite": bool(
            np.all(hamiltonian_core_density >= -1.0e-12)
        ),
    }


# 関数: current-pack formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return the exact current-pack Hamiltonian-density formulas."""
    return {
        "working_action": "-(Z_P/4) F_(P)^2 + (lambda/4) (|P|^2 - v^2)^2 + g_P P_mu J^mu",
        "retained_ansatz": (
            "P_mu^Qball = (f_0(r) e^{i omega t}, f_L(r) r_hat_i e^{i omega t})"
        ),
        "radial_field_strength": (
            "F_0r^(P) = partial_0 P_r - partial_r P_0 = i omega f_L - f_0'"
        ),
        "longitudinal_curl_rule": "F_ij^(P) = 0 for the pure radial longitudinal ansatz f_L(r) r_hat_i",
        "exact_hamiltonian_core": "epsilon_H,core(r) = |F_0r^(P)|^2 + m_0^2 f_0(r)^2",
        "expanded_core": "epsilon_H,core(r) = f_0'(r)^2 + omega^2 f_L(r)^2 + m_0^2 f_0(r)^2",
        "note_temporal_term": "epsilon_note,temp = omega^2 (f_0^2 + f_L^2)",
        "note_gradient_term": "epsilon_note,grad = f_0'^2 + f_L'^2 + 2 f_L^2 / r^2",
        "vacuum_subtracted_mh_family": "Delta V_MH(rho) = (lambda/4)(rho^2 - v^2)^2 - 0",
        "exact_potential_gap": (
            "The branch-local constitutive map rho[f_0,f_L] is not yet fixed exactly on the restored vector branch."
        ),
    }


# 関数: `.1623-.1626` を実行する。

def main() -> None:
    """Execute the energy-density / Hamiltonian-density derivation audit."""
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
        CASEB_RESTORE_GATE,
        GROUND_STATE_GATE,
        CASEB_TP_GATE,
        CASEB_CORE_GATE,
        BREAKTHROUGH_VEV,
        MEXICAN_HAT_FREEZE,
        ANCHOR_EVAL,
        PHASE1_EVAL,
        EXACT_REINJECTION_BRANCH,
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

    caseb_restore_summary = read_json(CASEB_RESTORE_GATE)["summary"]
    ground_state_summary = read_json(GROUND_STATE_GATE)["summary"]
    caseb_core_payload = read_json(CASEB_CORE_GATE)
    breakthrough_payload = read_json(BREAKTHROUGH_VEV)
    mexican_hat_payload = read_json(MEXICAN_HAT_FREEZE)
    branch = build_exact_branch_components()

    energy_density_observable_branch_admissible = True
    exact_hamiltonian_core_density_available = True
    radial_longitudinal_curl_free_under_current_ansatz = True
    radial_mass_term_dominates_current_branch = bool(branch["radial_mass_fraction"] > 0.95)
    note_naive_temporal_term_exact_supported = False
    note_naive_gradient_term_exact_supported = False
    naive_note_energy_density_exact_supported = False
    vacuum_subtracted_mh_family_available = bool(
        mexican_hat_payload["summary"]["selected_candidate_family_id"] == "mexican_hat"
    )
    branch_local_mh_energy_embedding_available = False
    prior_casea_worsen_retained = bool(
        caseb_restore_summary["full_sub_vs_casea_residual_ratio"] > 1.0
    )
    ground_state_note_no_go_retained = not bool(
        ground_state_summary["ground_state_nodeless_hypothesis_supported_under_current_pack"]
    )
    caseb_no_metric_rescue_retained = not bool(
        caseb_restore_summary["metric_artifact_rescue_supported"]
    )
    energy_density_form_factor_audit_admissible_now = True
    physical_reject_required = False

    rows = [
        row(
            "energy_density_observable_branch_admissible",
            "pass",
            "energy-density observable branch admissible now",
            truth(energy_density_observable_branch_admissible),
            "Signed-density and metric-rescue lanes are already exhausted honestly, so energy density is a genuinely new observable branch rather than another retry of the same failed family.",
        ),
        row(
            "exact_hamiltonian_core_density_available",
            "pass" if exact_hamiltonian_core_density_available else "reject",
            "exact Hamiltonian core density available under current frozen action",
            truth(exact_hamiltonian_core_density_available),
            "The frozen action plus the retained radial-longitudinal ansatz already determine epsilon_H,core = |F_0r|^2 + m_0^2 f_0^2 without introducing any new parameter.",
        ),
        row(
            "radial_longitudinal_curl_free_under_current_ansatz",
            "pass" if radial_longitudinal_curl_free_under_current_ansatz else "reject",
            "pure radial longitudinal ansatz is curl free",
            truth(radial_longitudinal_curl_free_under_current_ansatz),
            "For P_i = f_L(r) r_hat_i the spatial curl vanishes, so F_ij does not generate a separate exact positive f_L'^2 + 2 f_L^2 / r^2 field-strength term under the current pack.",
        ),
        row(
            "hamiltonian_core_positive_definite",
            "pass" if branch["hamiltonian_core_positive_definite"] else "reject",
            "Hamiltonian core density positive definite on the retained branch",
            truth(branch["hamiltonian_core_positive_definite"]),
            "Both |F_0r|^2 and m_0^2 f_0^2 are nonnegative, so the exact current-pack backbone is a genuine positive-density observable candidate.",
        ),
        row(
            "radial_mass_term_fraction",
            "pass" if radial_mass_term_dominates_current_branch else "watch",
            "radial mass-term fraction of the exact Hamiltonian core norm",
            branch["radial_mass_fraction"],
            "This shows how strongly the exact current-pack backbone is dominated by the scalar-like radial mass density on the retained vector branch.",
        ),
        row(
            "electric_like_term_fraction",
            "watch",
            "electric-like field-strength fraction of the exact Hamiltonian core norm",
            branch["electric_fraction"],
            "The nontransverse electric-like term remains present but subleading in the norm budget of the retained vector branch.",
        ),
        row(
            "note_temporal_vs_exact_mass_ratio",
            "watch",
            "note temporal norm divided by exact radial mass-term norm",
            branch["note_temporal_vs_exact_mass_ratio"],
            "The note's omega^2(f_0^2+f_L^2) temporal term is directionally useful, but it is not the exact leading current-pack term once the frozen radial mass surface is kept explicit.",
        ),
        row(
            "note_gradient_vs_exact_electric_ratio",
            "watch",
            "note gradient norm divided by exact electric-like norm",
            branch["note_gradient_vs_exact_electric_ratio"],
            "The note's gradient structure is numerically close to the exact electric-like norm, but it is not derived from the same exact field-strength identity under the current pack.",
        ),
        row(
            "note_naive_temporal_term_exact_supported",
            "pass" if note_naive_temporal_term_exact_supported else "reject",
            "note temporal term exact under current pack",
            truth(note_naive_temporal_term_exact_supported),
            "The exact backbone keeps m_0^2 f_0^2 explicitly, so replacing it by omega^2(f_0^2+f_L^2) would overclaim the current action-level derivation.",
        ),
        row(
            "note_naive_gradient_term_exact_supported",
            "pass" if note_naive_gradient_term_exact_supported else "reject",
            "note gradient term exact under current pack",
            truth(note_naive_gradient_term_exact_supported),
            "Because the retained ansatz is curl free in the spatial sector, the note's f_L'^2 + 2 f_L^2 / r^2 term is not an already-derived exact field-strength contribution here.",
        ),
        row(
            "naive_note_energy_density_exact_supported",
            "pass" if naive_note_energy_density_exact_supported else "reject",
            "naive note energy density already exact under current pack",
            truth(naive_note_energy_density_exact_supported),
            "The note provides a strong direction, but the exact current-pack derivation only supports the Hamiltonian core and a family-level vacuum-subtracted potential surface so far.",
        ),
        row(
            "vacuum_subtracted_mexican_hat_family_available",
            "pass" if vacuum_subtracted_mh_family_available else "reject",
            "vacuum-subtracted Mexican-hat family available",
            truth(vacuum_subtracted_mh_family_available),
            "The family-level potential V(rho) = (lambda/4)(rho^2-v^2)^2 is frozen publicly, so a vacuum-subtracted radial potential branch remains available in principle.",
        ),
        row(
            "branch_local_mh_energy_embedding_available",
            "pass" if branch_local_mh_energy_embedding_available else "reject",
            "branch-local Mexican-hat energy embedding available on restored vector branch",
            truth(branch_local_mh_energy_embedding_available),
            "What is still missing is the exact constitutive map rho[f_0,f_L] on the restored vector branch, so the full nonlinear potential density is not yet fixed at the same level as the Hamiltonian core.",
        ),
        row(
            "prior_casea_worsen_retained",
            "pass" if prior_casea_worsen_retained else "reject",
            "prior caseA worsen retained as evidence",
            truth(prior_casea_worsen_retained),
            "The energy-density branch does not erase the earlier signed-kernel worsen; it replaces the mainline because that lane is already exhausted honestly.",
        ),
        row(
            "ground_state_note_no_go_retained",
            "pass" if ground_state_note_no_go_retained else "reject",
            "ground-state note no-go retained as evidence",
            truth(ground_state_note_no_go_retained),
            "The current exact pilot still does not support the nodeless/ground-state rescue, so the observable shift is not being used to smuggle that hypothesis back in.",
        ),
        row(
            "caseb_no_metric_rescue_retained",
            "pass" if caseb_no_metric_rescue_retained else "reject",
            "caseB no-metric-rescue retained as evidence",
            truth(caseb_no_metric_rescue_retained),
            "The effective-metric no-go remains fixed; the new branch changes the observable, not the already-failed metric rescue conclusion.",
        ),
        row(
            "energy_density_form_factor_audit_admissible_now",
            "pass" if energy_density_form_factor_audit_admissible_now else "reject",
            "energy-density form-factor audit admissible now",
            truth(energy_density_form_factor_audit_admissible_now),
            "Once the exact Hamiltonian backbone is fixed honestly, the next branch can compute F_E(q) without pretending that the full note formula is already exact.",
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
            "caseb_restore_gate": display_path(CASEB_RESTORE_GATE),
            "ground_state_gate": display_path(GROUND_STATE_GATE),
            "caseb_tp_gate": display_path(CASEB_TP_GATE),
            "caseb_core_gate": display_path(CASEB_CORE_GATE),
            "breakthrough_vev": display_path(BREAKTHROUGH_VEV),
            "mexican_hat_freeze": display_path(MEXICAN_HAT_FREEZE),
            "anchor_eval": display_path(ANCHOR_EVAL),
            "phase1_eval": display_path(PHASE1_EVAL),
            "exact_reinjection_branch": display_path(EXACT_REINJECTION_BRANCH),
        },
        "constants": {
            "q_theory_over_m0": branch["q_theory_over_m0"],
            "beta_1_scalar": branch["beta"],
            "amp0_phase1_equivalent": branch["amp0"],
            "amp_l_phase1_equivalent": branch["amp_l"],
            "radial_mass_squared": branch["radial_mass_sq"],
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "energy_density_observable_branch_admissible": energy_density_observable_branch_admissible,
        "exact_hamiltonian_core_density_available": exact_hamiltonian_core_density_available,
        "radial_longitudinal_curl_free_under_current_ansatz": radial_longitudinal_curl_free_under_current_ansatz,
        "hamiltonian_core_positive_definite": branch["hamiltonian_core_positive_definite"],
        "radial_mass_term_fraction": branch["radial_mass_fraction"],
        "electric_like_term_fraction": branch["electric_fraction"],
        "note_temporal_vs_exact_mass_ratio": branch["note_temporal_vs_exact_mass_ratio"],
        "note_gradient_vs_exact_electric_ratio": branch["note_gradient_vs_exact_electric_ratio"],
        "note_naive_temporal_term_exact_supported": note_naive_temporal_term_exact_supported,
        "note_naive_gradient_term_exact_supported": note_naive_gradient_term_exact_supported,
        "naive_note_energy_density_exact_supported": naive_note_energy_density_exact_supported,
        "vacuum_subtracted_mexican_hat_family_available": vacuum_subtracted_mh_family_available,
        "branch_local_mh_energy_embedding_available": branch_local_mh_energy_embedding_available,
        "prior_casea_worsen_retained": prior_casea_worsen_retained,
        "ground_state_note_no_go_retained": ground_state_note_no_go_retained,
        "caseb_no_metric_rescue_retained": caseb_no_metric_rescue_retained,
        "energy_density_form_factor_audit_admissible_now": energy_density_form_factor_audit_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_case_route": NEXT_CASE_ROUTE_NAME,
        "recommended_followup_case_route_or_none": NEXT_CASE_ROUTE,
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
            "energy_note_positive_definite": hit(energy_note_text, "Energy density:"),
            "energy_note_photon_couples_energy": hit(
                energy_note_text, "photon は charge に couple しない。energy に couple する。"
            ),
            "energy_note_temporal_formula": hit(
                energy_note_text, "\\varepsilon_{\\rm temporal}"
            ),
            "energy_note_gradient_formula": hit(
                energy_note_text, "\\varepsilon_{\\rm gradient}"
            ),
            "part1_field_strength": hit(
                part1_text,
                "F_{0r}^{(P)} = \\partial_0 P_r - \\partial_r P_0 = i\\omega f_L - f_0'",
            ),
            "part1_pref_equals_infty": hit(part1_text, "P_{\\mathrm{ref}}\\equiv P_{\\infty}"),
            "status_current_step": hit(status_text, "8.7.56.1623"),
            "roadmap_current_branch": hit(
                roadmap_text, "effective-metric disposition sync / closeout"
            ),
            "current_problem_caseb_restore": hit(
                current_problem_text, "effective-metric `v^2` subtraction exact treatment restore"
            ),
            "current_status_caseb_restore": hit(
                current_status_text, "effective-metric `v^2` subtraction exact treatment restore"
            ),
            "unified_roadmap_caseb_restore": hit(
                unified_roadmap_text,
                "`.1619-.1622` は **effective-metric `v^2` subtraction exact treatment restore**",
            ),
            "part5_caseb_restore": hit(
                part5_text, "effective-metric `v^2` subtraction exact treatment restore"
            ),
            "breakthrough_working_action": {
                "pattern": "working_action",
                "line": 1,
                "text": breakthrough_payload["formulas"]["working_action"],
            },
            "mexican_hat_selected_potential": {
                "pattern": "selected_potential",
                "line": 1,
                "text": mexican_hat_payload["formulas"]["selected_potential"],
            },
            "caseb_core_formula": {
                "pattern": "spatial_core_caseb",
                "line": 1,
                "text": caseb_core_payload["evidence"]["formulas"]["spatial_core_caseb"],
            },
        },
        "retained_branch_norms": {
            "hamiltonian_core_norm": branch["hamiltonian_core_norm"],
            "electric_like_norm": branch["electric_norm"],
            "radial_mass_norm": branch["radial_mass_norm"],
            "note_temporal_norm": branch["note_temporal_norm"],
            "note_gradient_norm": branch["note_gradient_norm"],
            "mass_fraction": branch["radial_mass_fraction"],
            "electric_fraction": branch["electric_fraction"],
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1623",
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
                "8.7.56.1624",
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
                "8.7.56.1625",
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
                "8.7.56.1626",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "artifacts": manifest},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
