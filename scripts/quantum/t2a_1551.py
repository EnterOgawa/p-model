#!/usr/bin/env python3
"""Generate 8.7.56.1551-.1554 Q-ball / defect embedding audit artifacts.

This branch audits whether the restored exact vector / Q-ball branch can be
embedded into the defect collective-coordinate sector used by the already-passed
generic fermion-emergence / spinor-mapping route (`8.7.29.3`).

The audit is computation-first and strict:

- localization of the restored branch may justify a translation collective
  coordinate candidate,
- the explicit harmonic factor may justify a U(1)-phase coordinate candidate,
  but
- the generic defect spinor sector additionally requires an SU(2)-type internal
  orientation plus a topological/Hopf/FR block.

Without those missing blocks, the full Q-ball-to-defect embedding does not open,
so the effective source theorem must remain downstream.
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
    / "q_8_7_56_1547_1550_constitutive_map_reopen_declaration_gate_metrics.json"
)
PRIOR_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1547_1550_constitutive_map_reopen_route_sync_metrics.json"
)
ANCHOR_NUMERIC = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
)
ANCHOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_declaration_gate_metrics.json"
)
MATTER_EMBEDDING_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1539_1542_matter_rot_embedding_audit_declaration_gate_metrics.json"
)
MICRO_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1543_1546_micro_source_fn_deriv_declaration_gate_metrics.json"
)
SPINOR_AUDIT = ROOT / "output" / "public" / "quantum" / "fermion_emergence_spinor_mapping_audit.json"
JEFF_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_derivation_20260328.md")

STEP_TAG = "8.7.56.1551-1554"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor Q-ball / defect collective-coordinate embedding audit"
STEM = build_compact_artifact_stem(STEP_TAG, "qball_defect_embedding_audit", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_generic_spinor_emergence_available_qball_defect_embedding_gap_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_qball_defect_embedding_partial_translation_u1_only_internal_su2_hopf_gap_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_internal_su2_hopf_embedding_reopen_audit"
)
NEXT_ROUTE = "8.7.56.1555"


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
    """Return the defect-sector requirement and Q-ball branch formulas."""
    return {
        "generic_defect_sector": (
            "P(x,t) = P_defect(x-X(t), U(t)) + deltaP(x,t), U in SU(2), "
            "psi_defect = sqrt(rho_defect(x-X)) * chi(t)"
        ),
        "generic_topological_block": (
            "L_ext = |D_mu P|^2 - V(|P|) - 1/4 F_munu F^munu + lambda_H * J_Hopf[P]"
        ),
        "restored_exact_vector_branch": (
            "P_mu^Qball = (f_0(r)e^{i omega t}, f_L(r) rhat_i e^{i omega t})"
        ),
        "partial_embedding_only": (
            "{localized center shift X(t), harmonic U(1) phase} available, "
            "but SU(2) internal orientation / Hopf topological block missing"
        ),
    }


# 関数: `.1551-.1554` を実行する。

def main() -> None:
    """Execute the Q-ball / defect collective-coordinate embedding audit."""
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
        ANCHOR_GATE,
        MATTER_EMBEDDING_GATE,
        MICRO_GATE,
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
    anchor_gate_summary = read_json(ANCHOR_GATE)["summary"]
    matter_summary = read_json(MATTER_EMBEDDING_GATE)["summary"]
    micro_summary = read_json(MICRO_GATE)["summary"]
    spinor_audit = read_json(SPINOR_AUDIT)

    formulas = build_formulae()

    qball_pack_text = "\n".join(
        [
            json.dumps(anchor_summary, ensure_ascii=False),
            json.dumps(anchor_gate_summary, ensure_ascii=False),
            json.dumps(matter_summary, ensure_ascii=False),
            json.dumps(micro_summary, ensure_ascii=False),
            part1_text,
            part3a_text,
        ]
    )
    defect_pack_text = json.dumps(spinor_audit, ensure_ascii=False)

    qball_formula_hit = hit(current_problem_text, "P_\\mu^{\\rm Qball}")
    qball_formula_status_hit = hit(current_status_text, "P_\\mu^{\\rm Qball}")
    defect_formula_hit = hit(defect_pack_text, "P_defect(x - X(t), U(t))")
    defect_su2_hit = hit(defect_pack_text, "U in SU(2)")
    defect_hopf_hit = hit(defect_pack_text, "lambda_H * J_Hopf[P]")
    defect_fr_hit = hit(defect_pack_text, "pi4(M)=Z2")
    defect_spin_return_hit = hit(defect_pack_text, "U(2pi) psi_defect = -psi_defect")
    jeff_case1_hit = hit(jeff_note_text, "Case I: J_eff⁰ ≈ |f₀|²")

    prior_embedding_gap_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("qball_defect_embedding_audit_required", False)
        and not prior_summary.get("constitutive_map_opened_for_current_branch", True)
        and not prior_summary.get("effective_source_theorem_attempt_admissible_now", True)
    )

    localized_restored_exact_branch_available = bool(
        anchor_summary.get("anchor_preserving_continuation_restored", False)
        and anchor_summary.get("localized_until_phase1_equivalent", False)
        and anchor_summary.get("localized_across_full_sampled_path", False)
        and bool(anchor_summary.get("phase1_equivalent_row", {}).get("localized", False))
        and bool(anchor_summary.get("phase1_equivalent_row", {}).get("nontrivial", False))
    )
    translation_collective_coordinate_candidate_available = localized_restored_exact_branch_available
    harmonic_u1_phase_coordinate_available = bool(qball_formula_hit or qball_formula_status_hit)

    defect_sector_requires_su2_orientation = bool(
        defect_su2_hit and spinor_audit.get("scenario_applied_features", {}).get("selected_pi4_group") == "Z2"
    )
    defect_sector_requires_topological_block = bool(
        defect_hopf_hit and spinor_audit.get("scenario_applied_features", {}).get("selected_has_topological_term", False)
    )
    defect_sector_requires_fr_z2_spin_return = bool(defect_fr_hit and defect_spin_return_hit)

    explicit_su2_orientation_coordinate_available_on_qball_branch = count_hits(qball_pack_text, "U in SU(2)") > 0
    explicit_hopf_topological_block_available_on_qball_branch = (
        count_hits(qball_pack_text, "lambda_H * J_Hopf[P]") > 0
    )
    fr_z2_spinor_return_available_on_qball_branch = (
        count_hits(qball_pack_text, "pi4(M)=Z2") > 0
        or count_hits(qball_pack_text, "U(2pi) psi_defect = -psi_defect") > 0
    )

    qball_defect_embedding_partial_translation_u1_only = bool(
        translation_collective_coordinate_candidate_available
        and harmonic_u1_phase_coordinate_available
        and not explicit_su2_orientation_coordinate_available_on_qball_branch
        and not explicit_hopf_topological_block_available_on_qball_branch
        and not fr_z2_spinor_return_available_on_qball_branch
    )
    qball_defect_collective_coordinate_embedding_opened = False
    internal_su2_hopf_embedding_reopen_required = bool(
        prior_embedding_gap_ready
        and qball_defect_embedding_partial_translation_u1_only
        and defect_sector_requires_su2_orientation
        and defect_sector_requires_topological_block
        and defect_sector_requires_fr_z2_spin_return
        and not qball_defect_collective_coordinate_embedding_opened
    )
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False

    available_dynamic_collective_coordinate_count_proxy = 4.0
    required_defect_dynamic_collective_coordinate_count_proxy = 6.0
    dynamic_collective_coordinate_ratio_proxy = (
        required_defect_dynamic_collective_coordinate_count_proxy
        / available_dynamic_collective_coordinate_count_proxy
    )

    rows = [
        row(
            "prior_embedding_gap_ready",
            "pass" if prior_embedding_gap_ready else "reject",
            "prior Q-ball / defect embedding gap branch ready",
            truth(prior_embedding_gap_ready),
            "This audit only starts after `.1547-.1550` localized the blocker at the Q-ball-to-defect embedding gap.",
        ),
        row(
            "localized_restored_exact_branch_available",
            "pass" if localized_restored_exact_branch_available else "reject",
            "localized restored exact vector branch available",
            truth(localized_restored_exact_branch_available),
            "The anchor-preserving restored exact branch remains localized and nontrivial at the retained Phase-1 equivalent point.",
        ),
        row(
            "translation_collective_coordinate_candidate_available",
            "pass" if translation_collective_coordinate_candidate_available else "reject",
            "translation collective-coordinate candidate available",
            truth(translation_collective_coordinate_candidate_available),
            "Localization of the restored branch supports an inferred center-shift collective coordinate candidate X(t).",
        ),
        row(
            "harmonic_u1_phase_coordinate_available",
            "pass" if harmonic_u1_phase_coordinate_available else "reject",
            "harmonic U(1) phase coordinate available",
            truth(harmonic_u1_phase_coordinate_available),
            "The restored exact vector / Q-ball branch explicitly carries the harmonic factor e^{i omega t}.",
        ),
        row(
            "defect_sector_requires_su2_orientation",
            "pass" if defect_sector_requires_su2_orientation else "reject",
            "defect spinor sector requires SU(2) orientation",
            truth(defect_sector_requires_su2_orientation),
            "The generic fermion-emergence pass is written on a defect sector with U in SU(2).",
        ),
        row(
            "defect_sector_requires_topological_block",
            "pass" if defect_sector_requires_topological_block else "reject",
            "defect spinor sector requires topological block",
            truth(defect_sector_requires_topological_block),
            "The generic spinor-emergence chain explicitly uses lambda_H * J_Hopf[P].",
        ),
        row(
            "defect_sector_requires_fr_z2_spin_return",
            "pass" if defect_sector_requires_fr_z2_spin_return else "reject",
            "defect spinor sector requires FR Z2 spin return",
            truth(defect_sector_requires_fr_z2_spin_return),
            "The generic pass explicitly requires pi4(M)=Z2 and 2pi/4pi spinor return.",
        ),
        row(
            "explicit_su2_orientation_coordinate_available_on_qball_branch",
            "pass" if explicit_su2_orientation_coordinate_available_on_qball_branch else "reject",
            "explicit SU(2) orientation coordinate available on Q-ball branch",
            truth(explicit_su2_orientation_coordinate_available_on_qball_branch),
            "No current branch-local artifact identifies an SU(2)-type internal orientation coordinate on the restored exact vector / Q-ball branch.",
        ),
        row(
            "explicit_hopf_topological_block_available_on_qball_branch",
            "pass" if explicit_hopf_topological_block_available_on_qball_branch else "reject",
            "explicit Hopf/topological block available on Q-ball branch",
            truth(explicit_hopf_topological_block_available_on_qball_branch),
            "No current branch-local artifact carries a Hopf/WZ/Chern-type topological term for the restored exact vector / Q-ball branch.",
        ),
        row(
            "fr_z2_spinor_return_available_on_qball_branch",
            "pass" if fr_z2_spinor_return_available_on_qball_branch else "reject",
            "FR Z2 spinor return available on Q-ball branch",
            truth(fr_z2_spinor_return_available_on_qball_branch),
            "No current branch-local artifact identifies pi4(M)=Z2 or a 2pi/4pi spinor-return structure on the restored exact vector / Q-ball branch.",
        ),
        row(
            "available_dynamic_collective_coordinate_count_proxy",
            "pass",
            "available Q-ball dynamic collective-coordinate count proxy",
            available_dynamic_collective_coordinate_count_proxy,
            "The current branch gives translation (3) plus harmonic U(1) phase (1) as the strongest collective-coordinate candidates.",
        ),
        row(
            "required_defect_dynamic_collective_coordinate_count_proxy",
            "pass",
            "required defect dynamic collective-coordinate count proxy",
            required_defect_dynamic_collective_coordinate_count_proxy,
            "The generic defect spinor sector requires translation (3) plus SU(2)-orientation (3) before the extra topological block is counted.",
        ),
        row(
            "dynamic_collective_coordinate_ratio_proxy",
            "pass",
            "required-to-available dynamic collective-coordinate ratio proxy",
            dynamic_collective_coordinate_ratio_proxy,
            "This proxy shows that the restored branch only supplies a partial subset of the defect collective coordinates.",
        ),
        row(
            "qball_defect_embedding_partial_translation_u1_only",
            "pass" if qball_defect_embedding_partial_translation_u1_only else "reject",
            "Q-ball / defect embedding is partial translation+U(1) only",
            truth(qball_defect_embedding_partial_translation_u1_only),
            "The strongest honest read is a partial embedding with localization and harmonic phase, but without SU(2)/Hopf/FR structure.",
        ),
        row(
            "qball_defect_collective_coordinate_embedding_opened",
            "pass" if qball_defect_collective_coordinate_embedding_opened else "reject",
            "full Q-ball / defect collective-coordinate embedding opened",
            truth(qball_defect_collective_coordinate_embedding_opened),
            "The current pack still does not close the full defect collective-coordinate embedding on the restored exact vector branch.",
        ),
        row(
            "internal_su2_hopf_embedding_reopen_required",
            "pass" if internal_su2_hopf_embedding_reopen_required else "reject",
            "internal SU(2)/Hopf embedding reopen required",
            truth(internal_su2_hopf_embedding_reopen_required),
            "The next honest blocker is the missing internal-orientation / topological-sector embedding, not the source theorem itself.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Retrying the source theorem now would skip the still-missing SU(2)/Hopf defect-sector embedding.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable-dictionary work remains downstream of a successful exact current closure and source theorem.",
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
            "anchor_gate": display_path(ANCHOR_GATE),
            "matter_embedding_gate": display_path(MATTER_EMBEDDING_GATE),
            "micro_gate": display_path(MICRO_GATE),
            "spinor_audit": display_path(SPINOR_AUDIT),
            "jeff_derivation_note": display_path(JEFF_NOTE),
        },
        "prior_metrics": {
            "prior_problem_classification": prior_summary.get("trial2_numeric_alpha_problem_classification"),
            "anchor_problem_classification": anchor_summary.get("trial2_numeric_alpha_problem_classification"),
            "matter_problem_classification": matter_summary.get("trial2_numeric_alpha_problem_classification"),
            "micro_problem_classification": micro_summary.get("trial2_numeric_alpha_problem_classification"),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "localized_restored_exact_branch_available": localized_restored_exact_branch_available,
        "translation_collective_coordinate_candidate_available": translation_collective_coordinate_candidate_available,
        "harmonic_u1_phase_coordinate_available": harmonic_u1_phase_coordinate_available,
        "defect_sector_requires_su2_orientation": defect_sector_requires_su2_orientation,
        "defect_sector_requires_topological_block": defect_sector_requires_topological_block,
        "defect_sector_requires_fr_z2_spin_return": defect_sector_requires_fr_z2_spin_return,
        "explicit_su2_orientation_coordinate_available_on_qball_branch": explicit_su2_orientation_coordinate_available_on_qball_branch,
        "explicit_hopf_topological_block_available_on_qball_branch": explicit_hopf_topological_block_available_on_qball_branch,
        "fr_z2_spinor_return_available_on_qball_branch": fr_z2_spinor_return_available_on_qball_branch,
        "qball_defect_embedding_partial_translation_u1_only": qball_defect_embedding_partial_translation_u1_only,
        "qball_defect_collective_coordinate_embedding_opened": qball_defect_collective_coordinate_embedding_opened,
        "internal_su2_hopf_embedding_reopen_required": internal_su2_hopf_embedding_reopen_required,
        "effective_source_theorem_attempt_admissible_now": effective_source_theorem_attempt_admissible_now,
        "observable_dictionary_gate_admissible_now": observable_dictionary_gate_admissible_now,
        "available_dynamic_collective_coordinate_count_proxy": available_dynamic_collective_coordinate_count_proxy,
        "required_defect_dynamic_collective_coordinate_count_proxy": required_defect_dynamic_collective_coordinate_count_proxy,
        "dynamic_collective_coordinate_ratio_proxy": dynamic_collective_coordinate_ratio_proxy,
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
            "qball_formula_problem": qball_formula_hit,
            "qball_formula_status": qball_formula_status_hit,
            "defect_formula": defect_formula_hit,
            "defect_su2": defect_su2_hit,
            "defect_hopf": defect_hopf_hit,
            "defect_fr": defect_fr_hit,
            "defect_spin_return": defect_spin_return_hit,
            "jeff_case1": jeff_case1_hit,
        },
        "search_counts": {
            "qball_pack_su2_count": count_hits(qball_pack_text, "U in SU(2)"),
            "qball_pack_hopf_count": count_hits(qball_pack_text, "lambda_H * J_Hopf[P]"),
            "qball_pack_fr_count": count_hits(qball_pack_text, "pi4(M)=Z2"),
            "qball_pack_spin_return_count": count_hits(qball_pack_text, "U(2pi) psi_defect = -psi_defect"),
        },
        "carry_over": {
            "prior_summary": prior_summary,
            "prior_route_summary": prior_route_summary,
            "anchor_summary": anchor_summary,
            "matter_summary": matter_summary,
            "micro_summary": micro_summary,
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
        payload(
            "8.7.56.1551",
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
            "8.7.56.1552",
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
            "8.7.56.1553",
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
            "8.7.56.1554",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] Q-ball / defect embedding audit artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
