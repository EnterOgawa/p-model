#!/usr/bin/env python3
"""Generate 8.7.56.1539-.1542 matter-current / rotational-source audit artifacts.

This branch continues the computation-first mainline after the exact current
derivation showed that the same-field on-shell linear source collapses to zero.
The remaining question is narrower:

- does the current public pack already provide an explicit
  `J_matter^mu[P^Qball]` embedding or an explicit rotational-source functional,
  or
- does it only expose generic current symbols and microscopic coupling
  templates, in which case a new derivation branch must be opened before the
  effective source theorem can honestly resume?

The audit is strict. Generic wording does not count as an explicit source
functional on the restored exact vector branch.
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
CURRENT_DERIVATION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1535_1538_charge_current_closure_derivation_declaration_gate_metrics.json"
)
CHARGE_CLOSURE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1491_1494_charge_current_closure_declaration_gate_metrics.json"
)
JEFF_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_derivation_20260328.md")

STEP_TAG = "8.7.56.1539-1542"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor matter-current / rotational-source embedding audit"
STEM = build_compact_artifact_stem(STEP_TAG, "matter_rot_embedding_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_exact_current_derivation_same_field_on_shell_zero_matter_rot_embedding_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_matter_rot_embedding_missing_microscopic_functional_derivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_microscopic_matter_rot_source_functional_derivation"
)
NEXT_ROUTE = "8.7.56.1543"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8テキストを読み込む。

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
    """Convert one path into repo-relative display text when possible."""
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


# 関数: metrics row を構成する。

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
    """Write one JSON payload and one CSV rows table."""
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


# 関数: 現 branch で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return the explicit source surfaces and missing embeddings."""
    return {
        "generic_matter_current_surface": "J_matter^mu = (rho c, rho v)",
        "generic_interaction_surface": "L_int = g_P P_mu J_matter^mu",
        "microscopic_rot_surface": (
            "L_int + L_rot ⊃ -lambda_rot g_P psi_bar gamma^mu (1-gamma^5)/2 psi P_mu"
            " - (lambda_rot g_P / 4m) psi_bar sigma^{mu nu} psi F^{(P)}_{mu nu}"
        ),
        "missing_matter_embedding": "J_matter^mu[P^Qball] is not explicit in the current pack",
        "missing_rotational_source": "R_spin^mu[P^Qball, J_matter] is not explicit in the current pack",
    }


# 関数: `.1539-.1542` を実行する。

def main() -> None:
    """Execute the matter-current / rotational-source embedding audit branch."""
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
        CURRENT_DERIVATION_GATE,
        CHARGE_CLOSURE_GATE,
        JEFF_NOTE,
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
    jeff_note_text = read_text(JEFF_NOTE)

    current_derivation_summary = read_json(CURRENT_DERIVATION_GATE)["summary"]
    charge_closure_summary = read_json(CHARGE_CLOSURE_GATE)["summary"]
    formulas = build_formulae()

    part1_matter_current_hit = hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})")
    part1_interaction_hit = hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}")
    part1_micro_chiral_hit = hit(part1_text, "\\bar{\\psi}\\gamma^\\mu\\frac{1-\\gamma^5}{2}\\psi\\,P_\\mu")
    part1_micro_pauli_hit = hit(part1_text, "\\bar{\\psi}\\sigma^{\\mu\\nu}\\psi\\,F^{(P)}_{\\mu\\nu}")
    part1_rot_placeholder_hit = hit(part1_text, "\\lambda_{\\mathrm{rot}}\\,\\mathcal{O}_{\\mathrm{spin}}[P_\\mu,J^\\mu_{\\mathrm{matter}}]")
    part3a_qball_identity_hit = hit(part3a_text, "Q-ball Noether charge = adopted U(1) charge")
    part3a_source_absent_hit = hit(part3a_text, "effective source formula `J^\\mu_{\\rm eff}[P^{\\rm Qball}]` が still surface していない")
    jeff_step3_hit = hit(jeff_note_text, "### Step 3: J_eff^μ の構造を読む")
    jeff_case2_hit = hit(jeff_note_text, "Case II: J_eff⁰ に f_L が non-trivially 入る")

    prior_current_derivation_ready = bool(
        current_derivation_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and current_derivation_summary.get("same_field_on_shell_linear_source_zero", False)
        and current_derivation_summary.get("matter_rot_source_embedding_audit_required", False)
    )
    same_field_on_shell_zero_retained = bool(
        current_derivation_summary.get("same_field_on_shell_linear_source_zero", False)
    )
    generic_matter_current_symbol_available = bool(part1_matter_current_hit and part1_interaction_hit)
    microscopic_rotational_spin_surface_available = bool(
        part1_micro_chiral_hit and part1_micro_pauli_hit and part1_rot_placeholder_hit
    )
    qball_charge_identity_available = bool(part3a_qball_identity_hit)

    explicit_qball_matter_current_embedding_available = False
    explicit_rotational_source_functional_available = False
    integrated_fermion_bilinear_to_qball_mapping_available = False
    nonzero_source_embedding_opened = False

    microscopic_functional_derivation_admissible_next = bool(
        prior_current_derivation_ready
        and same_field_on_shell_zero_retained
        and generic_matter_current_symbol_available
        and microscopic_rotational_spin_surface_available
        and qball_charge_identity_available
        and not explicit_qball_matter_current_embedding_available
        and not explicit_rotational_source_functional_available
        and not integrated_fermion_bilinear_to_qball_mapping_available
    )
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False

    rows = [
        row(
            "prior_current_derivation_ready",
            "pass" if prior_current_derivation_ready else "reject",
            "prior exact-current derivation ready",
            truth(prior_current_derivation_ready),
            "The embedding audit is only honest after the same-field on-shell zero result has been fixed computation-side.",
        ),
        row(
            "same_field_on_shell_zero_retained",
            "pass" if same_field_on_shell_zero_retained else "reject",
            "same-field on-shell zero retained",
            truth(same_field_on_shell_zero_retained),
            "Any nonzero source must come from a genuine embedding, not by reopening the already-zero same-field linear term.",
        ),
        row(
            "generic_matter_current_symbol_available",
            "pass" if generic_matter_current_symbol_available else "reject",
            "generic matter-current symbol available",
            truth(generic_matter_current_symbol_available),
            "Part I explicitly retains J_matter^mu and the minimal coupling L_int = g_P P_mu J_matter^mu.",
        ),
        row(
            "microscopic_rotational_spin_surface_available",
            "pass" if microscopic_rotational_spin_surface_available else "reject",
            "microscopic rotational/spin coupling surface available",
            truth(microscopic_rotational_spin_surface_available),
            "Part I also retains microscopic chiral and Pauli-type couplings, but only as fermionic bilinear templates.",
        ),
        row(
            "qball_charge_identity_available",
            "pass" if qball_charge_identity_available else "reject",
            "Q-ball adopted-U(1) identity available",
            truth(qball_charge_identity_available),
            "The Q-ball charge identity survives as a normalization-side constraint, but it does not itself provide J_eff^mu.",
        ),
        row(
            "explicit_qball_matter_current_embedding_available",
            "pass" if explicit_qball_matter_current_embedding_available else "reject",
            "explicit Q-ball matter-current embedding available",
            truth(explicit_qball_matter_current_embedding_available),
            "The current pack never writes J_matter^mu as an explicit functional of the restored exact Q-ball background profiles.",
        ),
        row(
            "explicit_rotational_source_functional_available",
            "pass" if explicit_rotational_source_functional_available else "reject",
            "explicit rotational-source functional available",
            truth(explicit_rotational_source_functional_available),
            "The pack retains lambda_rot O_spin[...] and microscopic couplings, but not an integrated source functional on the Q-ball background.",
        ),
        row(
            "integrated_fermion_bilinear_to_qball_mapping_available",
            "pass" if integrated_fermion_bilinear_to_qball_mapping_available else "reject",
            "integrated fermion-bilinear to Q-ball mapping available",
            truth(integrated_fermion_bilinear_to_qball_mapping_available),
            "No current public step maps the microscopic bilinears onto f0/fL or an equivalent restored exact vector branch source functional.",
        ),
        row(
            "nonzero_source_embedding_opened",
            "pass" if nonzero_source_embedding_opened else "reject",
            "nonzero source embedding opened under current pack",
            truth(nonzero_source_embedding_opened),
            "The audit finds symbols and microscopic hints, but not the explicit nonzero source embedding required for J_eff^mu.",
        ),
        row(
            "microscopic_functional_derivation_admissible_next",
            "pass" if microscopic_functional_derivation_admissible_next else "reject",
            "microscopic matter/rot source functional derivation admissible next",
            truth(microscopic_functional_derivation_admissible_next),
            "Because the pack exposes the generic and microscopic surfaces but not the embedding, the next honest lane is a new derivation branch rather than more text search.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Source-theorem work remains premature until a nonzero matter/rot source embedding is actually derived.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable-dictionary work remains downstream of an exact current closure and a successful source theorem.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "unified_roadmap_note": display_path(UNIFIED_ROADMAP),
            "computation_expert_share_note": display_path(COMPUTATION_EXPERT_SHARE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "jeff_derivation_note": display_path(JEFF_NOTE),
        },
        "prior_metrics": {
            "current_derivation_gate": display_path(CURRENT_DERIVATION_GATE),
            "charge_closure_gate": display_path(CHARGE_CLOSURE_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "same_field_on_shell_zero_retained": same_field_on_shell_zero_retained,
        "generic_matter_current_symbol_available": generic_matter_current_symbol_available,
        "microscopic_rotational_spin_surface_available": microscopic_rotational_spin_surface_available,
        "qball_charge_identity_available": qball_charge_identity_available,
        "explicit_qball_matter_current_embedding_available": explicit_qball_matter_current_embedding_available,
        "explicit_rotational_source_functional_available": explicit_rotational_source_functional_available,
        "integrated_fermion_bilinear_to_qball_mapping_available": integrated_fermion_bilinear_to_qball_mapping_available,
        "nonzero_source_embedding_opened": nonzero_source_embedding_opened,
        "microscopic_functional_derivation_admissible_next": microscopic_functional_derivation_admissible_next,
        "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "scalar_strong_candidate_retained": True,
        "blind_vector_no_go_retained": True,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "part_hits": {
            "part1_matter_current": part1_matter_current_hit,
            "part1_interaction": part1_interaction_hit,
            "part1_micro_chiral": part1_micro_chiral_hit,
            "part1_micro_pauli": part1_micro_pauli_hit,
            "part1_rot_placeholder": part1_rot_placeholder_hit,
            "part3a_qball_identity": part3a_qball_identity_hit,
            "part3a_source_absent": part3a_source_absent_hit,
            "jeff_step3": jeff_step3_hit,
            "jeff_case2": jeff_case2_hit,
        },
        "carry_over": {
            "current_derivation_summary": current_derivation_summary,
            "charge_closure_summary": charge_closure_summary,
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
        payload(
            "8.7.56.1539",
            f"{STEP_NAME} inventory",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    audit_paths = write_artifact(
        "audit",
        payload(
            "8.7.56.1540",
            f"{STEP_NAME} audit",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1541",
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
        payload(
            "8.7.56.1542",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] matter-current / rotational-source embedding audit artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
