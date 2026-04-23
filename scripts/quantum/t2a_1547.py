#!/usr/bin/env python3
"""Generate 8.7.56.1547-.1550 constitutive-map reopen audit artifacts.

This branch reopens the constitutive-map question after `.1543-.1546` fixed
that Part III-A's `psi <-> P` bridge is only a scalar-envelope / Noether-energy
proxy. The new question is narrower:

- does the already-passed generic fermion-emergence / spinor-mapping surface
  (`8.7.29.3`) supply the missing constitutive map for the restored exact
  vector / Q-ball branch, or
- is that spinor-emergence result still a distinct defect-collective-coordinate
  sector, leaving a Q-ball-to-defect embedding gap that must be audited before
  the source theorem can honestly resume?

The audit is strict. A generic spinor-emergence pass only counts if it is
explicitly tied to the restored exact vector branch profiles or their Q-ball
background variables.
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
MICRO_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1543_1546_micro_source_fn_deriv_declaration_gate_metrics.json"
)
EMBEDDING_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1539_1542_matter_rot_embedding_audit_declaration_gate_metrics.json"
)
SPINOR_AUDIT = ROOT / "output" / "public" / "quantum" / "fermion_emergence_spinor_mapping_audit.json"
JEFF_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_derivation_20260328.md")

STEP_TAG = "8.7.56.1547-1550"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor scalar-envelope / spinor-bilinear constitutive-map reopen audit"
STEM = build_compact_artifact_stem(STEP_TAG, "constitutive_map_reopen", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_microscopic_functional_derivation_failed_scalar_to_spinor_constitutive_gap_reopen_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_generic_spinor_emergence_available_qball_defect_embedding_gap_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_qball_defect_collective_coordinate_embedding_audit"
)
NEXT_ROUTE = "8.7.56.1551"


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


# 関数: branch で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return the generic spinor-emergence and missing branch map formulae."""
    return {
        "generic_spinor_emergence_chain": (
            "P(x,t) = P_defect(x-X(t), U(t)) + deltaP(x,t), U in SU(2),"
            " psi_defect = sqrt(rho_defect(x-X)) * chi(t)"
        ),
        "generic_dirac_like_operator": (
            "i d_t psi_defect = [v alpha·(-i nabla) + m beta + V_defect(x-X)] psi_defect"
        ),
        "restored_exact_vector_branch": (
            "P_mu^Qball = (f_0(r)e^{i omega t}, f_L(r) rhat_i e^{i omega t})"
        ),
        "missing_embedding": (
            "{f_0, f_L, lambda_scale, restored exact vector branch} -> {P_defect, X(t), U(t), rho_defect, chi(t)}"
        ),
    }


# 関数: `.1547-.1550` を実行する。

def main() -> None:
    """Execute the constitutive-map reopen audit branch."""
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
        MICRO_GATE,
        EMBEDDING_GATE,
        SPINOR_AUDIT,
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

    micro_summary = read_json(MICRO_GATE)["summary"]
    embedding_summary = read_json(EMBEDDING_GATE)["summary"]
    spinor_audit = read_json(SPINOR_AUDIT)
    formulas = build_formulae()

    part1_micro_chiral_hit = hit(part1_text, "\\bar{\\psi}\\gamma^\\mu\\frac{1-\\gamma^5}{2}\\psi")
    part1_micro_pauli_hit = hit(part1_text, "\\bar{\\psi}\\sigma^{\\mu\\nu}\\psi")
    part3a_psi_bridge_hit = hit(part3a_text, "\\delta P_{+}/P_{*}")
    spinor_audit_pass = bool(spinor_audit.get("decision", {}).get("spinor_mapping_hard_pass", False))
    spinor_audit_scenario = str(spinor_audit.get("scenario", {}).get("name") or "")
    spinor_chain_hit = hit(json.dumps(spinor_audit, ensure_ascii=False), "psi_defect")
    qball_branch_hit = hit(part5_text, "restored exact vector / Q-ball branch")
    jeff_case1_hit = hit(jeff_note_text, "Case I: J_eff⁰ ≈ |f₀|²")

    prior_micro_gap_ready = bool(
        micro_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and micro_summary.get("constitutive_map_reopen_required", False)
        and not micro_summary.get("nonzero_source_embedding_opened", True)
    )
    embedding_gap_retained = bool(
        not embedding_summary.get("explicit_qball_matter_current_embedding_available", True)
        and not embedding_summary.get("explicit_rotational_source_functional_available", True)
    )

    generic_spinor_emergence_surface_available = bool(
        spinor_audit_pass and spinor_audit_scenario == "defect_operator_extension" and spinor_chain_hit
    )
    generic_internal_spinor_operator_available = bool(
        spinor_audit.get("defect_operator_metrics", {}).get("operator_closure_pass", False)
    )
    generic_external_spinor_dependency_closed = bool(
        not spinor_audit.get("scenario_applied_features", {}).get("selected_external_spinor_dependency", True)
    )

    qball_branch_to_defect_collective_coordinate_map_available = False
    restored_exact_vector_branch_to_internal_spinor_map_available = False
    microscopic_bilinear_closure_available_on_qball_branch = False
    constitutive_map_opened_for_current_branch = False

    qball_channel_count = 4.0
    defect_collective_coordinate_count = 7.0
    qball_to_defect_coordinate_ratio = defect_collective_coordinate_count / qball_channel_count

    qball_defect_embedding_audit_required = bool(
        prior_micro_gap_ready
        and embedding_gap_retained
        and generic_spinor_emergence_surface_available
        and generic_internal_spinor_operator_available
        and generic_external_spinor_dependency_closed
        and not qball_branch_to_defect_collective_coordinate_map_available
        and not restored_exact_vector_branch_to_internal_spinor_map_available
        and not microscopic_bilinear_closure_available_on_qball_branch
        and not constitutive_map_opened_for_current_branch
    )
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False

    rows = [
        row(
            "prior_micro_gap_ready",
            "pass" if prior_micro_gap_ready else "reject",
            "prior microscopic constitutive-gap branch ready",
            truth(prior_micro_gap_ready),
            "This audit only starts after `.1543-.1546` localized the blocker at the scalar-to-spinor constitutive gap.",
        ),
        row(
            "embedding_gap_retained",
            "pass" if embedding_gap_retained else "reject",
            "Q-ball matter/rot embedding gap retained",
            truth(embedding_gap_retained),
            "The branch-specific Q-ball matter-current and rotational-source embeddings are still absent and remain part of the blocker.",
        ),
        row(
            "generic_spinor_emergence_surface_available",
            "pass" if generic_spinor_emergence_surface_available else "reject",
            "generic spinor-emergence surface available",
            truth(generic_spinor_emergence_surface_available),
            "Step 8.7.29.3 does give a public defect-collective-coordinate spinor-emergence pass.",
        ),
        row(
            "generic_internal_spinor_operator_available",
            "pass" if generic_internal_spinor_operator_available else "reject",
            "generic internal spinor operator available",
            truth(generic_internal_spinor_operator_available),
            "The defect-operator extension supplies an internal first-order spinor operator without external spinor dependency.",
        ),
        row(
            "generic_external_spinor_dependency_closed",
            "pass" if generic_external_spinor_dependency_closed else "reject",
            "generic external spinor dependency closed",
            truth(generic_external_spinor_dependency_closed),
            "The generic spinor-emergence audit closes external spinor dependency at the defect-collective-coordinate level.",
        ),
        row(
            "qball_branch_to_defect_collective_coordinate_map_available",
            "pass" if qball_branch_to_defect_collective_coordinate_map_available else "reject",
            "Q-ball branch to defect collective-coordinate map available",
            truth(qball_branch_to_defect_collective_coordinate_map_available),
            "The current pack never identifies the restored exact vector / Q-ball branch with the P_defect(X,U) sector used by the generic spinor-emergence pass.",
        ),
        row(
            "restored_exact_vector_branch_to_internal_spinor_map_available",
            "pass" if restored_exact_vector_branch_to_internal_spinor_map_available else "reject",
            "restored exact vector branch to internal spinor map available",
            truth(restored_exact_vector_branch_to_internal_spinor_map_available),
            "No explicit map sends {f0, fL, lambda_scale} to psi_defect or its collective coordinates.",
        ),
        row(
            "microscopic_bilinear_closure_available_on_qball_branch",
            "pass" if microscopic_bilinear_closure_available_on_qball_branch else "reject",
            "microscopic bilinear closure available on Q-ball branch",
            truth(microscopic_bilinear_closure_available_on_qball_branch),
            "Without a Q-ball-to-defect embedding, the generic spinor-emergence pass does not yet close the microscopic bilinears on the current branch.",
        ),
        row(
            "qball_real_channel_count",
            "pass",
            "restored exact vector branch real channel count proxy",
            qball_channel_count,
            "The restored exact vector branch currently tracks temporal and longitudinal profiles, i.e. four real profile channels before extra collective coordinates.",
        ),
        row(
            "defect_collective_coordinate_count",
            "pass",
            "defect collective-coordinate count proxy",
            defect_collective_coordinate_count,
            "The generic spinor-emergence surface uses position plus SU(2)-type collective coordinates beyond the Q-ball profile data.",
        ),
        row(
            "qball_to_defect_coordinate_ratio",
            "pass",
            "defect-to-Q-ball coordinate ratio proxy",
            qball_to_defect_coordinate_ratio,
            "This proxy highlights that the generic defect map requires extra collective coordinates not yet embedded into the restored exact vector branch.",
        ),
        row(
            "constitutive_map_opened_for_current_branch",
            "pass" if constitutive_map_opened_for_current_branch else "reject",
            "constitutive map opened for current branch",
            truth(constitutive_map_opened_for_current_branch),
            "The generic spinor-emergence pass is not yet sufficient to reopen the constitutive map on the current Q-ball branch.",
        ),
        row(
            "qball_defect_embedding_audit_required",
            "pass" if qball_defect_embedding_audit_required else "reject",
            "Q-ball / defect embedding audit required",
            truth(qball_defect_embedding_audit_required),
            "The next honest lane is to audit whether the restored exact vector / Q-ball branch can be embedded into the defect-collective-coordinate spinor sector.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Retrying the source theorem now would skip the still-missing Q-ball-to-defect constitutive embedding.",
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
            "spinor_audit": display_path(SPINOR_AUDIT),
            "jeff_derivation_note": display_path(JEFF_NOTE),
        },
        "prior_metrics": {
            "micro_gate": display_path(MICRO_GATE),
            "embedding_gate": display_path(EMBEDDING_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "generic_spinor_emergence_surface_available": generic_spinor_emergence_surface_available,
        "generic_internal_spinor_operator_available": generic_internal_spinor_operator_available,
        "generic_external_spinor_dependency_closed": generic_external_spinor_dependency_closed,
        "qball_branch_to_defect_collective_coordinate_map_available": qball_branch_to_defect_collective_coordinate_map_available,
        "restored_exact_vector_branch_to_internal_spinor_map_available": restored_exact_vector_branch_to_internal_spinor_map_available,
        "microscopic_bilinear_closure_available_on_qball_branch": microscopic_bilinear_closure_available_on_qball_branch,
        "constitutive_map_opened_for_current_branch": constitutive_map_opened_for_current_branch,
        "qball_defect_embedding_audit_required": qball_defect_embedding_audit_required,
        "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "qball_real_channel_count": qball_channel_count,
        "defect_collective_coordinate_count": defect_collective_coordinate_count,
        "qball_to_defect_coordinate_ratio": qball_to_defect_coordinate_ratio,
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
            "part1_micro_chiral": part1_micro_chiral_hit,
            "part1_micro_pauli": part1_micro_pauli_hit,
            "part3a_psi_bridge": part3a_psi_bridge_hit,
            "spinor_audit_chain": spinor_chain_hit,
            "part5_qball_branch": qball_branch_hit,
            "jeff_case1": jeff_case1_hit,
        },
        "carry_over": {
            "micro_summary": micro_summary,
            "embedding_summary": embedding_summary,
            "spinor_audit_summary": {
                "overall_status": spinor_audit.get("decision", {}).get("overall_status"),
                "spinor_mapping_hard_pass": spinor_audit.get("decision", {}).get("spinor_mapping_hard_pass"),
                "scenario": spinor_audit.get("scenario", {}).get("name"),
                "selected_external_spinor_dependency": spinor_audit.get("scenario_applied_features", {}).get("selected_external_spinor_dependency"),
                "operator_closure_pass": spinor_audit.get("defect_operator_metrics", {}).get("operator_closure_pass"),
            },
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
            "8.7.56.1547",
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
            "8.7.56.1548",
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
            "8.7.56.1549",
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
            "8.7.56.1550",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] constitutive-map reopen audit artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
