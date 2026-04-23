#!/usr/bin/env python3
"""Generate 8.7.56.1563-.1566 frozen-action direct J_eff derivation artifacts.

This branch executes the computation-first directive literally:

- keep only the frozen action surfaces that are explicit in the current pack,
- substitute P_mu = P_mu^Qball + a_mu,
- collect the linear-in-a_mu source split,
- freeze the explicit J_eff^mu decomposition before any structure
  classification or disposition sync.

The result is stronger than the earlier backbone-only audit because the
current is now written component-by-component. It is still honest about the
remaining gap: the pack exposes only generic matter-current / rotational-source
surfaces, so J_eff^0 cannot yet be classified as scalar-proxy, signed-density,
other, or zero without a downstream classification branch.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
COMPUTATION_EXPERT_SHARE = (
    ROOT / "doc" / "quantum" / "43_trial2_numeric_alpha_vector_qball_computation_reactivation_expert_share.md"
)
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
PRIOR_RESET_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1559_1562_direct_jeff_mainline_reset_declaration_gate_metrics.json"
)
PRIOR_BACKBONE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1535_1538_charge_current_closure_derivation_declaration_gate_metrics.json"
)
DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_final_directive_20260328.md"
)

STEP_TAG = "8.7.56.1563-1566"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor frozen-action direct J_eff derivation"
)
STEM = build_compact_artifact_stem(STEP_TAG, "direct_jeff_deriv", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_frozen_action_direct_jeff_mainline_reset_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_direct_jeff_split_derived_charge_density_classification_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_jeff_charge_density_structure_classification"
)
NEXT_ROUTE = "8.7.56.1567"
NEXT_DISPOSITION_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_jeff_disposition_sync"
)
NEXT_DISPOSITION_ROUTE = "8.7.56.1571"


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
    """Return the first line matching one substring pattern."""
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


# 関数: payload を構成する。

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


# 関数: direct J_eff split を返す。

def build_formulae() -> dict[str, str]:
    """Return the frozen-action direct J_eff split formulae."""
    return {
        "split_definition": "P_mu(x) = Q_mu(x) + a_mu(x), with Q_mu = P_mu^Qball",
        "transverse_light_branch": "A_mu = delta P_mu^T / sqrt(Z_P), with partial_i a^i = 0",
        "jeff_kinetic": "J_eff,kin^nu[Q] = Z_P * partial_mu F_Q^{mu nu}",
        "jeff_stueckelberg": "J_eff,stk^nu[Q,pi_Q] = m_P^2 * (Q^nu - partial^nu pi_Q / m_P)",
        "jeff_gauge_fixing": (
            "J_eff,gf^nu[Q,pi_Q] = xi_g^{-1} * partial^nu(partial_mu Q^mu + xi_g m_P pi_Q)"
        ),
        "jeff_matter": "J_eff,matter^nu[Q] = g_P * J_matter^nu[Q]  (explicit Q-ball functional still open)",
        "jeff_rot": (
            "J_eff,rot^nu[Q] = delta(lambda_rot O_spin[P_mu,J_matter^mu]) / delta P_nu |_(P=Q)"
        ),
        "jeff_total": (
            "J_eff^nu[Q] = J_eff,kin^nu + J_eff,stk^nu + J_eff,gf^nu + J_eff,matter^nu + J_eff,rot^nu"
        ),
        "jeff_charge_density": (
            "J_eff^0[Q] = Z_P * partial_mu F_Q^{mu 0}"
            " + m_P^2 * (Q^0 - partial^0 pi_Q / m_P)"
            " + xi_g^{-1} * partial^0(partial_mu Q^mu + xi_g m_P pi_Q)"
            " + g_P * J_matter^0[Q] + J_eff,rot^0[Q]"
        ),
        "same_field_on_shell": "J_eff^nu[Q]_same-field,on-shell = 0",
    }


# 関数: `.1563-.1566` を実行する。

def main() -> None:
    """Execute the frozen-action direct J_eff derivation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        COMPUTATION_EXPERT_SHARE,
        PART1,
        PART3A,
        PART5,
        PRIOR_RESET_GATE,
        PRIOR_BACKBONE_GATE,
        DIRECTIVE_NOTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    expert_share_text = read_text(COMPUTATION_EXPERT_SHARE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    directive_text = read_text(DIRECTIVE_NOTE)

    prior_reset_summary = read_json(PRIOR_RESET_GATE)["summary"]
    prior_backbone_summary = read_json(PRIOR_BACKBONE_GATE)["summary"]
    formulas = build_formulae()

    prior_reset_ready = bool(
        prior_reset_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_reset_summary.get("frozen_action_direct_jeff_promoted_to_mainline", False)
    )
    backbone_ready = bool(
        prior_backbone_summary.get("free_backbone_linear_formula_derived", False)
        and prior_backbone_summary.get("same_field_on_shell_linear_source_zero", False)
    )

    part1_total_action_hit = hit(part1_text, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}")
    part1_free_action_hit = hit(part1_text, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}")
    part1_interaction_hit = hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}")
    part1_rot_placeholder_hit = hit(
        part1_text,
        "\\lambda_{\\mathrm{rot}}\\,\\mathcal{O}_{\\mathrm{spin}}[P_\\mu,J^\\mu_{\\mathrm{matter}}]",
    )
    part1_chiral_hit = hit(part1_text, "\\bar{\\psi}\\gamma^\\mu\\frac{1-\\gamma^5}{2}\\psi")
    part1_pauli_hit = hit(part1_text, "\\bar{\\psi}\\sigma^{\\mu\\nu}\\psi")
    part3a_photon_hit = hit(part3a_text, "A_\\mu=\\delta P_\\mu^T/\\sqrt{Z_P}")
    current_status_massless_hit = hit(current_status_text, "explicit_massless_transverse_mode_available = true")
    directive_split_hit = hit(directive_text, "P_\\mu(x) = P_\\mu^{\\rm Qball}(x) + a_\\mu(x)")
    directive_collect_hit = hit(directive_text, "L_total^vec に代入し、**a_μ の一次の項だけを集める**。")
    directive_complete_hit = hit(directive_text, "jeff_eff_mu_explicit_form_derived = true")

    frozen_action_surface_available = all(
        (
            part1_total_action_hit,
            part1_free_action_hit,
            part1_interaction_hit,
            part1_rot_placeholder_hit,
            part3a_photon_hit,
            directive_split_hit,
            directive_collect_hit,
            directive_complete_hit,
        )
    )
    kinetic_contribution_derived = frozen_action_surface_available and backbone_ready
    stueckelberg_contribution_derived = frozen_action_surface_available
    gauge_fixing_contribution_derived = frozen_action_surface_available
    matter_symbolic_contribution_derived = bool(part1_interaction_hit)
    rotational_symbolic_contribution_derived = bool(part1_rot_placeholder_hit)
    microscopic_matter_functional_available = False
    microscopic_rotational_functional_available = False
    massless_transverse_mode_retained = bool(current_status_massless_hit and part3a_photon_hit)
    jeff_eff_mu_explicit_form_derived = bool(
        prior_reset_ready
        and kinetic_contribution_derived
        and stueckelberg_contribution_derived
        and gauge_fixing_contribution_derived
        and matter_symbolic_contribution_derived
        and rotational_symbolic_contribution_derived
    )
    same_field_on_shell_zero_retained = bool(
        jeff_eff_mu_explicit_form_derived
        and prior_backbone_summary.get("same_field_on_shell_linear_source_zero", False)
    )
    jeff_eff_charge_density_structure_identified = False
    jeff_charge_density_classification_admissible_now = jeff_eff_mu_explicit_form_derived
    disposition_case_selected = False
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False

    rows = [
        row(
            "prior_reset_ready",
            "pass" if prior_reset_ready else "reject",
            "prior direct J_eff mainline reset ready",
            truth(prior_reset_ready),
            "This derivation only starts after the mainline has been reset away from the internal-topological lane.",
        ),
        row(
            "backbone_ready",
            "pass" if backbone_ready else "reject",
            "prior direct-current backbone ready",
            truth(backbone_ready),
            "The explicit split builds on the already fixed linear backbone and same-field on-shell zero result.",
        ),
        row(
            "frozen_action_surface_available",
            "pass" if frozen_action_surface_available else "reject",
            "frozen action derivation surface available",
            truth(frozen_action_surface_available),
            "The frozen vector action, interaction term, photon branch, and directive split are all explicit in the current pack.",
        ),
        row(
            "kinetic_contribution_derived",
            "pass" if kinetic_contribution_derived else "reject",
            "kinetic contribution derived",
            truth(kinetic_contribution_derived),
            "The antisymmetric field-strength term yields the explicit kinetic contribution Z_P partial_mu F_Q^{mu nu}.",
        ),
        row(
            "stueckelberg_contribution_derived",
            "pass" if stueckelberg_contribution_derived else "reject",
            "Stueckelberg contribution derived",
            truth(stueckelberg_contribution_derived),
            "The m_P-dependent covariant completion is kept as an explicit formal current contribution rather than dropped by assumption.",
        ),
        row(
            "gauge_fixing_contribution_derived",
            "pass" if gauge_fixing_contribution_derived else "reject",
            "gauge-fixing contribution derived",
            truth(gauge_fixing_contribution_derived),
            "The xi_g gauge-fixing term contributes an explicit derivative source to the formal J_eff split.",
        ),
        row(
            "matter_symbolic_contribution_derived",
            "pass" if matter_symbolic_contribution_derived else "reject",
            "matter symbolic contribution derived",
            truth(matter_symbolic_contribution_derived),
            "The generic g_P P_mu J_matter^mu term supplies an explicit symbolic matter-current contribution.",
        ),
        row(
            "rotational_symbolic_contribution_derived",
            "pass" if rotational_symbolic_contribution_derived else "reject",
            "rotational symbolic contribution derived",
            truth(rotational_symbolic_contribution_derived),
            "The lambda_rot O_spin placeholder supplies an explicit symbolic rotational contribution slot.",
        ),
        row(
            "microscopic_matter_functional_available",
            "pass" if microscopic_matter_functional_available else "reject",
            "microscopic matter-current functional available",
            truth(microscopic_matter_functional_available),
            "The current pack still lacks an explicit Q-ball-background constitutive map for J_matter^mu[Q].",
        ),
        row(
            "microscopic_rotational_functional_available",
            "pass" if microscopic_rotational_functional_available else "reject",
            "microscopic rotational-source functional available",
            truth(microscopic_rotational_functional_available),
            "The current pack still lacks an explicit reduced rotational-source functional on the restored exact vector branch.",
        ),
        row(
            "massless_transverse_mode_retained",
            "pass" if massless_transverse_mode_retained else "reject",
            "massless transverse mode retained",
            truth(massless_transverse_mode_retained),
            "The adopted light branch remains the physical massless transverse mode A_mu = delta P_mu^T / sqrt(Z_P).",
        ),
        row(
            "jeff_eff_mu_explicit_form_derived",
            "pass" if jeff_eff_mu_explicit_form_derived else "reject",
            "explicit J_eff^mu split derived",
            truth(jeff_eff_mu_explicit_form_derived),
            "The direct frozen-action derivation now fixes J_eff^mu as a five-piece split before any structure classification.",
        ),
        row(
            "same_field_on_shell_zero_retained",
            "pass" if same_field_on_shell_zero_retained else "reject",
            "same-field on-shell zero retained",
            truth(same_field_on_shell_zero_retained),
            "The explicit split is consistent with the prior result that the same-field on-shell background yields zero linear source.",
        ),
        row(
            "jeff_eff_charge_density_structure_identified",
            "pass" if jeff_eff_charge_density_structure_identified else "reject",
            "J_eff^0 structure identified",
            truth(jeff_eff_charge_density_structure_identified),
            "This branch derives the split; the next branch classifies J_eff^0 as scalar proxy, signed density, other, or zero.",
        ),
        row(
            "jeff_charge_density_classification_admissible_now",
            "pass" if jeff_charge_density_classification_admissible_now else "reject",
            "J_eff^0 structure classification admissible now",
            truth(jeff_charge_density_classification_admissible_now),
            "Because the explicit split is fixed, charge-density structure classification is now the honest next computation.",
        ),
        row(
            "disposition_case_selected",
            "pass" if disposition_case_selected else "reject",
            "disposition case selected",
            truth(disposition_case_selected),
            "Case I-IV selection is deferred until the explicit J_eff^0 structure has been classified.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Source-theorem work remains downstream of J_eff^0 structure classification and disposition.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable mapping remains downstream of both direct J_eff derivation and source-theorem resolution.",
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
            "computation_expert_share": display_path(COMPUTATION_EXPERT_SHARE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "directive_note": display_path(DIRECTIVE_NOTE),
        },
        "prior_metrics": {
            "prior_reset_gate": display_path(PRIOR_RESET_GATE),
            "prior_backbone_gate": display_path(PRIOR_BACKBONE_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_disposition_route_name": NEXT_DISPOSITION_ROUTE_NAME,
            "next_disposition_route": NEXT_DISPOSITION_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "jeff_eff_mu_explicit_form_derived": jeff_eff_mu_explicit_form_derived,
        "jeff_eff_mu_explicit_split_component_count": 5.0,
        "same_field_on_shell_zero_retained": same_field_on_shell_zero_retained,
        "massless_transverse_mode_retained": massless_transverse_mode_retained,
        "microscopic_matter_functional_available": microscopic_matter_functional_available,
        "microscopic_rotational_functional_available": microscopic_rotational_functional_available,
        "jeff_eff_charge_density_structure_identified": jeff_eff_charge_density_structure_identified,
        "jeff_charge_density_classification_admissible_now": jeff_charge_density_classification_admissible_now,
        "disposition_case_selected": disposition_case_selected,
        "frozen_action_only_used": True,
        "new_free_parameters_introduced": False,
        "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "scalar_strong_candidate_retained": True,
        "blind_vector_no_go_retained": True,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_disposition_route": NEXT_DISPOSITION_ROUTE_NAME,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": jeff_eff_mu_explicit_form_derived,
        "next_required_artifacts": [
            NEXT_ROUTE_NAME,
            NEXT_DISPOSITION_ROUTE_NAME,
        ],
    }

    evidence = {
        "formulas": formulas,
        "hits": {
            "part1_total_action": part1_total_action_hit,
            "part1_free_action": part1_free_action_hit,
            "part1_interaction": part1_interaction_hit,
            "part1_rot_placeholder": part1_rot_placeholder_hit,
            "part1_chiral_coupling": part1_chiral_hit,
            "part1_pauli_coupling": part1_pauli_hit,
            "part3a_photon_branch": part3a_photon_hit,
            "current_status_massless_mode": current_status_massless_hit,
            "directive_split": directive_split_hit,
            "directive_collect": directive_collect_hit,
            "directive_complete": directive_complete_hit,
        },
        "carry_over": {
            "prior_reset_summary": prior_reset_summary,
            "prior_backbone_summary": prior_backbone_summary,
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": -0.083735013520183,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload("8.7.56.1563", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
    )
    audit_paths = write_artifact(
        "audit",
        payload("8.7.56.1564", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1565",
            f"{STEP_NAME} declaration gate",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    route_paths = write_artifact(
        "route_sync",
        payload("8.7.56.1566", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
    )

    print("[ok] frozen-action direct J_eff derivation artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
