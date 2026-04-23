#!/usr/bin/env python3
"""Generate 8.7.56.1555-.1558 internal SU(2)/Hopf reopen audit artifacts.

This branch asks a narrower question than `.1551-.1554`:

- can the restored exact vector / Q-ball branch be extended from the already
  retained translation+U(1) partial embedding to a full defect-sector
  embedding,
- specifically by reopening the internal SU(2)-orientation sector,
  Hopf/topological block, and FR Z2 spin-return structure?

The audit is computation-first and strict. It rejects over-reading symbol-level
false positives such as a generic unitary `U(t)` or a nonlinearity coefficient
named `lambda_H` when these do not realize the required defect-sector
functionals.
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
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1551_1554_qball_defect_embedding_audit_declaration_gate_metrics.json"
)
PRIOR_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1551_1554_qball_defect_embedding_audit_route_sync_metrics.json"
)
ANCHOR_NUMERIC = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
)
SPINOR_AUDIT = ROOT / "output" / "public" / "quantum" / "fermion_emergence_spinor_mapping_audit.json"
JEFF_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_derivation_20260328.md")

STEP_TAG = "8.7.56.1555-1558"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor internal SU(2) / Hopf embedding reopen audit"
STEM = build_compact_artifact_stem(STEP_TAG, "su2_hopf_reopen_audit", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_qball_defect_embedding_partial_translation_u1_only_internal_su2_hopf_gap_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_internal_su2_hopf_reopen_failed_microscopic_internal_topological_functional_derivation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_internal_orientation_topological_functional_derivation"
)
NEXT_ROUTE = "8.7.56.1559"


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
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列の最初の一致行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return one first matching line for a substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 部分文字列の出現回数を返す。

def count_hits(text: str, pattern: str) -> int:
    """Count one substring occurrence."""
    return text.count(pattern)


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
    """Build one standard metrics payload."""
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


# 関数: branch で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas for the defect-side internal sector and branch-local reads."""
    return {
        "defect_internal_sector": (
            "L_eff[X,U] = M_eff/2 * Xdot^2 + i*kappa*Tr(U^dagger d_t U sigma3) - H_eff[X,U]"
        ),
        "defect_topological_block": (
            "L_ext = |D_mu P|^2 - V(|P|) - 1/4 F_munu F^munu + lambda_H * J_Hopf[P]"
        ),
        "defect_spin_return": (
            "U(2pi) psi_defect = -psi_defect, U(4pi) psi_defect = psi_defect"
        ),
        "restored_exact_vector_branch": (
            "P_mu^Qball = (f_0(r)e^{i omega t}, f_L(r) rhat_i e^{i omega t})"
        ),
        "false_positive_unitary_symbol": (
            "U(t) (rho tensor rho_D0 tensor rho_E0) U^dagger(t) [generic unitary evolution, not defect SU(2) orientation]"
        ),
        "false_positive_lambda_h_symbol": (
            "lambda_H^(N0) [nonlinearity coefficient, not Hopf coupling lambda_H * J_Hopf[P]]"
        ),
    }


# 関数: `.1555-.1558` を実行する。

def main() -> None:
    """Execute the internal SU(2) / Hopf embedding reopen audit."""
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
        PRIOR_GATE,
        PRIOR_ROUTE,
        ANCHOR_NUMERIC,
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

    prior_summary = read_json(PRIOR_GATE)["summary"]
    prior_route_summary = read_json(PRIOR_ROUTE)["summary"]
    anchor_summary = read_json(ANCHOR_NUMERIC)["summary"]
    spinor_audit = read_json(SPINOR_AUDIT)
    formulas = build_formulae()

    qball_pack_text = "\n".join(
        [
            status_text,
            roadmap_text,
            current_problem_text,
            current_status_text,
            unified_roadmap_text,
            expert_share_text,
            part1_text,
            part3a_text,
            part5_text,
            json.dumps(anchor_summary, ensure_ascii=False),
            json.dumps(prior_summary, ensure_ascii=False),
        ]
    )
    defect_pack_text = json.dumps(spinor_audit, ensure_ascii=False)

    defect_internal_hit = hit(defect_pack_text, "i*kappa*Tr(U^dagger d_t U sigma3)")
    defect_hopf_hit = hit(defect_pack_text, "lambda_H * J_Hopf[P]")
    defect_fr_hit = hit(defect_pack_text, "pi4(M)=Z2")
    defect_spin_return_hit = hit(defect_pack_text, "U(2pi) psi_defect = -psi_defect")
    false_positive_unitary_hit = hit(part3a_text, "U(t)")
    false_positive_lambda_hit = hit(part1_text, "\\lambda_H^{(N0)}")
    jeff_charge_current_hit = hit(jeff_note_text, "J_eff^μ")

    prior_partial_embedding_gap_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("qball_defect_embedding_partial_translation_u1_only", False)
        and not prior_summary.get("qball_defect_collective_coordinate_embedding_opened", True)
        and prior_summary.get("internal_su2_hopf_embedding_reopen_required", False)
    )

    defect_internal_orientation_functional_required = bool(defect_internal_hit)
    defect_topological_block_required = bool(defect_hopf_hit)
    defect_fr_spin_return_required = bool(defect_fr_hit and defect_spin_return_hit)

    branch_local_internal_orientation_functional_available = (
        count_hits(qball_pack_text, "i*kappa*Tr(U^dagger d_t U sigma3)") > 0
    )
    branch_local_hopf_functional_available = count_hits(qball_pack_text, "lambda_H * J_Hopf[P]") > 0
    branch_local_fr_spin_return_available = (
        count_hits(qball_pack_text, "pi4(M)=Z2") > 0
        or count_hits(qball_pack_text, "U(2pi) psi_defect = -psi_defect") > 0
    )

    part3a_unitary_symbol_present = bool(false_positive_unitary_hit)
    part3a_unitary_symbol_is_defect_orientation_functional = False
    part1_lambda_h_symbol_present = bool(false_positive_lambda_hit)
    part1_lambda_h_symbol_is_hopf_functional = False

    internal_orientation_false_positive_symbol_only = bool(
        part3a_unitary_symbol_present and not branch_local_internal_orientation_functional_available
    )
    hopf_false_positive_symbol_only = bool(
        part1_lambda_h_symbol_present and not branch_local_hopf_functional_available
    )

    opened_internal_structure_block_count = 0.0
    required_internal_structure_block_count = 3.0
    internal_structure_open_fraction = (
        opened_internal_structure_block_count / required_internal_structure_block_count
    )
    false_positive_symbol_count = float(
        int(internal_orientation_false_positive_symbol_only) + int(hopf_false_positive_symbol_only)
    )

    internal_su2_hopf_embedding_reopen_opened = False
    internal_su2_hopf_embedding_reopen_failed_honest = bool(
        prior_partial_embedding_gap_ready
        and defect_internal_orientation_functional_required
        and defect_topological_block_required
        and defect_fr_spin_return_required
        and not branch_local_internal_orientation_functional_available
        and not branch_local_hopf_functional_available
        and not branch_local_fr_spin_return_available
        and not internal_su2_hopf_embedding_reopen_opened
    )
    microscopic_internal_topological_functional_derivation_required = (
        internal_su2_hopf_embedding_reopen_failed_honest
    )
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False

    rows = [
        row(
            "prior_partial_embedding_gap_ready",
            "pass" if prior_partial_embedding_gap_ready else "reject",
            "prior partial translation+U(1) embedding gap branch ready",
            truth(prior_partial_embedding_gap_ready),
            "This audit only starts after `.1551-.1554` fixed the branch as partial translation+U(1) only.",
        ),
        row(
            "defect_internal_orientation_functional_required",
            "pass" if defect_internal_orientation_functional_required else "reject",
            "defect sector requires explicit internal orientation functional",
            truth(defect_internal_orientation_functional_required),
            "The retained defect-side spinor map explicitly uses i*kappa*Tr(U^dagger d_t U sigma3).",
        ),
        row(
            "defect_topological_block_required",
            "pass" if defect_topological_block_required else "reject",
            "defect sector requires explicit Hopf/topological block",
            truth(defect_topological_block_required),
            "The retained defect-side spinor map explicitly uses lambda_H * J_Hopf[P].",
        ),
        row(
            "defect_fr_spin_return_required",
            "pass" if defect_fr_spin_return_required else "reject",
            "defect sector requires FR Z2 spin return",
            truth(defect_fr_spin_return_required),
            "The retained defect-side spinor map explicitly requires pi4(M)=Z2 and 2pi/4pi spinor return.",
        ),
        row(
            "branch_local_internal_orientation_functional_available",
            "pass" if branch_local_internal_orientation_functional_available else "reject",
            "branch-local internal orientation functional available",
            truth(branch_local_internal_orientation_functional_available),
            "No current branch-local artifact supplies a defect-style SU(2)-orientation kinetic functional.",
        ),
        row(
            "branch_local_hopf_functional_available",
            "pass" if branch_local_hopf_functional_available else "reject",
            "branch-local Hopf/topological functional available",
            truth(branch_local_hopf_functional_available),
            "No current branch-local artifact supplies a Hopf/WZ/Chern block for the restored exact vector / Q-ball branch.",
        ),
        row(
            "branch_local_fr_spin_return_available",
            "pass" if branch_local_fr_spin_return_available else "reject",
            "branch-local FR spin return available",
            truth(branch_local_fr_spin_return_available),
            "No current branch-local artifact identifies a pi4(M)=Z2 / 2pi-to-4pi spin-return sector on the restored branch.",
        ),
        row(
            "part3a_unitary_symbol_present",
            "pass" if part3a_unitary_symbol_present else "reject",
            "generic unitary U(t) symbol present",
            truth(part3a_unitary_symbol_present),
            "Part III-A contains a generic measurement-evolution unitary U(t), which must not be over-read as defect SU(2) orientation.",
        ),
        row(
            "part3a_unitary_symbol_is_defect_orientation_functional",
            "pass" if part3a_unitary_symbol_is_defect_orientation_functional else "reject",
            "generic unitary U(t) closes defect orientation functional",
            truth(part3a_unitary_symbol_is_defect_orientation_functional),
            "The Part III-A U(t) is a false positive for this branch because it is not the defect collective-coordinate orientation functional.",
        ),
        row(
            "part1_lambda_h_symbol_present",
            "pass" if part1_lambda_h_symbol_present else "reject",
            "Part I lambda_H symbol present",
            truth(part1_lambda_h_symbol_present),
            "Part I contains lambda_H^(N0), but this is a nonlinearity coefficient rather than a Hopf coupling.",
        ),
        row(
            "part1_lambda_h_symbol_is_hopf_functional",
            "pass" if part1_lambda_h_symbol_is_hopf_functional else "reject",
            "Part I lambda_H symbol closes Hopf functional",
            truth(part1_lambda_h_symbol_is_hopf_functional),
            "The retained lambda_H^(N0) symbol is a false positive and does not realize lambda_H * J_Hopf[P].",
        ),
        row(
            "internal_orientation_false_positive_symbol_only",
            "pass" if internal_orientation_false_positive_symbol_only else "reject",
            "internal orientation over-read is false-positive only",
            truth(internal_orientation_false_positive_symbol_only),
            "The generic U(t) symbol is present but does not provide the required defect internal-orientation functional.",
        ),
        row(
            "hopf_false_positive_symbol_only",
            "pass" if hopf_false_positive_symbol_only else "reject",
            "Hopf over-read is false-positive only",
            truth(hopf_false_positive_symbol_only),
            "The lambda_H^(N0) symbol is present but does not provide the required Hopf/topological block.",
        ),
        row(
            "opened_internal_structure_block_count",
            "pass",
            "opened internal structure block count",
            opened_internal_structure_block_count,
            "The restored exact vector / Q-ball branch currently opens none of the three missing internal-sector blocks.",
        ),
        row(
            "required_internal_structure_block_count",
            "pass",
            "required internal structure block count",
            required_internal_structure_block_count,
            "The defect-side reopening requires internal orientation, topological block, and FR spin-return structure.",
        ),
        row(
            "internal_structure_open_fraction",
            "pass",
            "opened-to-required internal structure fraction",
            internal_structure_open_fraction,
            "This fraction quantifies that the missing internal-sector reopening remains completely unopened.",
        ),
        row(
            "false_positive_symbol_count",
            "pass",
            "false positive symbol count",
            false_positive_symbol_count,
            "Two tempting symbols remain available, but neither one realizes the required defect-sector functional.",
        ),
        row(
            "internal_su2_hopf_embedding_reopen_opened",
            "pass" if internal_su2_hopf_embedding_reopen_opened else "reject",
            "internal SU(2)/Hopf embedding reopen opened",
            truth(internal_su2_hopf_embedding_reopen_opened),
            "The current pack still does not reopen the missing internal orientation / topological sector on the restored branch.",
        ),
        row(
            "internal_su2_hopf_embedding_reopen_failed_honest",
            "pass" if internal_su2_hopf_embedding_reopen_failed_honest else "reject",
            "internal SU(2)/Hopf reopen fails honestly under current pack",
            truth(internal_su2_hopf_embedding_reopen_failed_honest),
            "The honest blocker is the missing microscopic internal-orientation / topological-sector functional derivation.",
        ),
        row(
            "microscopic_internal_topological_functional_derivation_required",
            "pass" if microscopic_internal_topological_functional_derivation_required else "reject",
            "microscopic internal/topological functional derivation required",
            truth(microscopic_internal_topological_functional_derivation_required),
            "The next honest lane is a new derivation branch, not a premature source-theorem retry.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Retrying the source theorem now would skip the still-missing internal SU(2)/Hopf embedding layer.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable-dictionary work remains downstream of exact current closure and source theorem.",
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
            "prior_gate": display_path(PRIOR_GATE),
            "prior_route": display_path(PRIOR_ROUTE),
            "anchor_numeric": display_path(ANCHOR_NUMERIC),
            "spinor_audit": display_path(SPINOR_AUDIT),
            "jeff_derivation_note": display_path(JEFF_NOTE),
        },
        "prior_metrics": {
            "prior_problem_classification": prior_summary.get("trial2_numeric_alpha_problem_classification"),
            "prior_next_route": prior_summary.get("recommended_next_route_or_none"),
            "anchor_problem_classification": anchor_summary.get("trial2_numeric_alpha_problem_classification"),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "defect_internal_orientation_functional_required": defect_internal_orientation_functional_required,
        "defect_topological_block_required": defect_topological_block_required,
        "defect_fr_spin_return_required": defect_fr_spin_return_required,
        "branch_local_internal_orientation_functional_available": branch_local_internal_orientation_functional_available,
        "branch_local_hopf_functional_available": branch_local_hopf_functional_available,
        "branch_local_fr_spin_return_available": branch_local_fr_spin_return_available,
        "part3a_unitary_symbol_present": part3a_unitary_symbol_present,
        "part3a_unitary_symbol_is_defect_orientation_functional": part3a_unitary_symbol_is_defect_orientation_functional,
        "part1_lambda_h_symbol_present": part1_lambda_h_symbol_present,
        "part1_lambda_h_symbol_is_hopf_functional": part1_lambda_h_symbol_is_hopf_functional,
        "internal_orientation_false_positive_symbol_only": internal_orientation_false_positive_symbol_only,
        "hopf_false_positive_symbol_only": hopf_false_positive_symbol_only,
        "opened_internal_structure_block_count": opened_internal_structure_block_count,
        "required_internal_structure_block_count": required_internal_structure_block_count,
        "internal_structure_open_fraction": internal_structure_open_fraction,
        "false_positive_symbol_count": false_positive_symbol_count,
        "internal_su2_hopf_embedding_reopen_opened": internal_su2_hopf_embedding_reopen_opened,
        "internal_su2_hopf_embedding_reopen_failed_honest": internal_su2_hopf_embedding_reopen_failed_honest,
        "microscopic_internal_topological_functional_derivation_required": microscopic_internal_topological_functional_derivation_required,
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
        "hits": {
            "defect_internal_functional": defect_internal_hit,
            "defect_hopf": defect_hopf_hit,
            "defect_fr": defect_fr_hit,
            "defect_spin_return": defect_spin_return_hit,
            "part3a_unitary_symbol": false_positive_unitary_hit,
            "part1_lambda_h_symbol": false_positive_lambda_hit,
            "jeff_charge_current": jeff_charge_current_hit,
        },
        "search_counts": {
            "qball_pack_internal_orientation_count": count_hits(qball_pack_text, "i*kappa*Tr(U^dagger d_t U sigma3)"),
            "qball_pack_hopf_count": count_hits(qball_pack_text, "lambda_H * J_Hopf[P]"),
            "qball_pack_fr_count": count_hits(qball_pack_text, "pi4(M)=Z2"),
            "qball_pack_spin_return_count": count_hits(qball_pack_text, "U(2pi) psi_defect = -psi_defect"),
            "part3a_unitary_symbol_count": count_hits(part3a_text, "U(t)"),
            "part1_lambda_h_symbol_count": count_hits(part1_text, "\\lambda_H^{(N0)}"),
        },
        "carry_over": {
            "prior_summary": prior_summary,
            "prior_route_summary": prior_route_summary,
            "anchor_summary": anchor_summary,
            "spinor_audit_summary": {
                "overall_status": spinor_audit.get("decision", {}).get("overall_status"),
                "spinor_mapping_hard_pass": spinor_audit.get("decision", {}).get("spinor_mapping_hard_pass"),
                "selected_pi4_group": spinor_audit.get("scenario_applied_features", {}).get("selected_pi4_group"),
                "selected_has_topological_term": spinor_audit.get("scenario_applied_features", {}).get("selected_has_topological_term"),
                "selected_external_spinor_dependency": spinor_audit.get("scenario_applied_features", {}).get("selected_external_spinor_dependency"),
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
        payload("8.7.56.1555", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
    )
    audit_paths = write_artifact(
        "audit",
        payload("8.7.56.1556", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload("8.7.56.1557", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
    )
    route_paths = write_artifact(
        "route_sync",
        payload("8.7.56.1558", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
    )

    print("[ok] internal SU(2) / Hopf reopen audit artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
