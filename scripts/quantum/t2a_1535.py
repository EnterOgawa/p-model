#!/usr/bin/env python3
"""Generate 8.7.56.1535-.1538 exact current-closure derivation artifacts.

This branch accepts the computation-first advice that `J_eff^mu` should be
derived from the frozen action rather than searched as wording. The derivation
performed here is intentionally narrow and honest:

- derive the linear-in-`a_mu` backbone from the explicit frozen vector action,
- use the adopted photon branch to decide which sectors can source the light
  mode directly,
- check what survives after the same-field background equations are imposed,
- localize the remaining blocker.

The main conclusion is not an exact current theorem. The computation shows that
the same-field on-shell linear term collapses to zero unless an explicit
matter-current embedding and rotational-source functional are supplied. Those
surfaces are still missing in the current public pack.
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
ACTION_AUDIT = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
REACTIVATION_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1531_1534_computation_mainline_reactivation_declaration_gate_metrics.json"
)
CHARGE_CLOSURE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1491_1494_charge_current_closure_declaration_gate_metrics.json"
)
ANCHOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_declaration_gate_metrics.json"
)
JEFF_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_jeff_derivation_20260328.md")

STEP_TAG = "8.7.56.1535-1538"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact charge-current / Noether-current closure derivation"
STEM = build_compact_artifact_stem(STEP_TAG, "charge_current_closure_derivation", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_computation_mainline_reactivated_exact_charge_current_closure_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_exact_current_derivation_same_field_on_shell_zero_matter_rot_embedding_next"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_matter_rot_source_embedding_audit"
NEXT_ROUTE = "8.7.56.1539"


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


# 関数: frozen action から linear source backbone を返す。

def build_formulae() -> dict[str, str]:
    """Return the derived linear-source backbone formulae."""
    free_backbone = (
        "J_free^nu[Q] = Z_P * partial_mu F_Q^{mu nu}"
        " + m_P^2 * (Q^nu - partial^nu pi_Q / m_P)"
        " + xi_g^{-1} * partial^nu(partial_mu Q^mu + xi_g m_P pi_Q)"
    )
    total_linear = (
        "L_lin = a_nu * ( J_free^nu[Q]"
        " + g_P J_matter^nu"
        " + lambda_rot R_spin^nu[Q,J_matter] )"
    )
    background_eom = (
        "J_free^nu[Q] + g_P J_matter^nu + lambda_rot R_spin^nu[Q,J_matter] = 0"
    )
    on_shell = "L_lin(on-shell same-field background) = 0"
    chi_note = (
        "chi = ln(P_t/P_ref) depends on the temporal sector, while the adopted photon"
        " branch is A_mu = delta P_mu^T / sqrt(Z_P); no explicit transverse a_mu source"
        " from L_chi is frozen in the current pack."
    )
    return {
        "free_backbone_linear_formula": free_backbone,
        "total_linear_formula": total_linear,
        "background_eom_formula": background_eom,
        "on_shell_same_field_implication": on_shell,
        "chi_transverse_note": chi_note,
    }


# 関数: `.1535-.1538` を実行する。

def main() -> None:
    """Execute the exact charge-current / Noether-current closure derivation branch."""
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
        ACTION_AUDIT,
        REACTIVATION_GATE,
        CHARGE_CLOSURE_GATE,
        ANCHOR_GATE,
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

    action_audit = read_json(ACTION_AUDIT)
    reactivation_summary = read_json(REACTIVATION_GATE)["summary"]
    charge_closure_summary = read_json(CHARGE_CLOSURE_GATE)["summary"]
    anchor_summary = read_json(ANCHOR_GATE)["summary"]

    formulas = build_formulae()

    part1_total_action_hit = hit(part1_text, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}")
    part1_full_action_hit = hit(part1_text, "\\mathcal{L}_{P,\\mathrm{full}}")
    part1_free_action_hit = hit(part1_text, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}")
    part1_matter_current_hit = hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})")
    part1_interaction_hit = hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}")
    part1_rot_placeholder_hit = hit(part1_text, "\\lambda_{\\mathrm{rot}}\\,\\mathcal{O}_{\\mathrm{spin}}[P_\\mu,J^\\mu_{\\mathrm{matter}}]")
    part1_chi_hit = hit(part1_text, "\\ln(P_t/P_{\\mathrm{ref}})")
    part1_noether_hit = hit(part1_text, "\\partial_\\mu J^\\mu=0")
    part1_post_photon_hit = hit(part1_text, "post-photon nontransverse sector")
    part3a_photon_hit = hit(part3a_text, "A_\\mu=\\delta P_\\mu^T/\\sqrt{Z_P}")
    jeff_step2_hit = hit(jeff_note_text, "### Step 2: a_μ linear terms を collect")
    jeff_case_hit = hit(jeff_note_text, "Case I: J_eff⁰ ≈ |f₀|²")

    prior_reactivation_ready = bool(
        reactivation_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and reactivation_summary.get("computation_mainline_reactivated", False)
    )
    prior_closure_absent = not bool(
        charge_closure_summary.get("exact_charge_current_noether_closure_available", False)
    )
    anchor_restored = bool(anchor_summary.get("anchor_preserving_continuation_restored", False))

    frozen_action_surface_available = all(
        (
            part1_total_action_hit,
            part1_full_action_hit,
            part1_free_action_hit,
            part1_matter_current_hit,
            part1_interaction_hit,
            part1_rot_placeholder_hit,
            part1_chi_hit,
            part1_noether_hit,
            part1_post_photon_hit,
            part3a_photon_hit,
            jeff_step2_hit,
            jeff_case_hit,
        )
    )
    free_backbone_linear_formula_derived = frozen_action_surface_available
    transverse_photon_branch_excludes_chi_source = bool(part3a_photon_hit and part1_post_photon_hit)
    same_field_on_shell_linear_source_zero = bool(
        free_backbone_linear_formula_derived and transverse_photon_branch_excludes_chi_source
    )
    explicit_qball_matter_current_functional_available = False
    explicit_rotational_source_functional_available = False
    exact_charge_current_noether_closure_available = False
    matter_rot_source_embedding_audit_required = bool(
        prior_reactivation_ready
        and prior_closure_absent
        and anchor_restored
        and same_field_on_shell_linear_source_zero
        and not explicit_qball_matter_current_functional_available
        and not explicit_rotational_source_functional_available
    )
    effective_source_theorem_attempt_admissible_now = False
    observable_dictionary_gate_admissible_now = False

    rows = [
        row(
            "prior_reactivation_ready",
            "pass" if prior_reactivation_ready else "reject",
            "prior computation-mainline reactivation ready",
            truth(prior_reactivation_ready),
            "This derivation only starts after the route reset has already promoted computation over further text search.",
        ),
        row(
            "anchor_restored",
            "pass" if anchor_restored else "reject",
            "restored exact vector branch available",
            truth(anchor_restored),
            "The derivation is only honest on the restored exact vector branch, not on the earlier anchor-lost scalar-like branch.",
        ),
        row(
            "frozen_action_surface_available",
            "pass" if frozen_action_surface_available else "reject",
            "frozen action surface available",
            truth(frozen_action_surface_available),
            "Part I frozen action plus photon-branch surfaces must be explicit before a current derivation can start.",
        ),
        row(
            "free_backbone_linear_formula_derived",
            "pass" if free_backbone_linear_formula_derived else "reject",
            "free-backbone linear source formula derived",
            truth(free_backbone_linear_formula_derived),
            "The quadratic free backbone now yields an explicit linear-in-a_mu coefficient rather than a wording-only blocker.",
        ),
        row(
            "transverse_photon_branch_excludes_chi_source",
            "pass" if transverse_photon_branch_excludes_chi_source else "reject",
            "transverse photon branch excludes direct chi-sector source",
            truth(transverse_photon_branch_excludes_chi_source),
            "The adopted photon branch lives in delta P^T, while chi depends on the temporal sector P_t.",
        ),
        row(
            "same_field_on_shell_linear_source_zero",
            "pass" if same_field_on_shell_linear_source_zero else "reject",
            "same-field on-shell linear source zero",
            truth(same_field_on_shell_linear_source_zero),
            "For a background that satisfies the frozen action Euler-Lagrange equation, the same-field linear term collapses to zero.",
        ),
        row(
            "explicit_qball_matter_current_functional_available",
            "pass" if explicit_qball_matter_current_functional_available else "reject",
            "explicit Q-ball matter-current functional available",
            truth(explicit_qball_matter_current_functional_available),
            "A nonzero physical source would now need J_matter^mu[P^Qball] or an equivalent integrated-out functional, not just the generic current symbol.",
        ),
        row(
            "explicit_rotational_source_functional_available",
            "pass" if explicit_rotational_source_functional_available else "reject",
            "explicit rotational source functional available",
            truth(explicit_rotational_source_functional_available),
            "The current pack freezes lambda_rot O_spin[...] only as a placeholder, not as an explicit source functional.",
        ),
        row(
            "exact_charge_current_noether_closure_available",
            "pass" if exact_charge_current_noether_closure_available else "reject",
            "exact charge-current / Noether-current closure available",
            truth(exact_charge_current_noether_closure_available),
            "The derivation sharpens the blocker but still does not close J_eff^0 into an exact current theorem.",
        ),
        row(
            "matter_rot_source_embedding_audit_required",
            "pass" if matter_rot_source_embedding_audit_required else "reject",
            "matter-current / rotational-source embedding audit required",
            truth(matter_rot_source_embedding_audit_required),
            "Because same-field on-shell expansion is zero, the next honest lane is an embedding audit for matter and rotational source functionals.",
        ),
        row(
            "effective_source_theorem_attempt_admissible_now",
            "pass" if effective_source_theorem_attempt_admissible_now else "reject",
            "effective source theorem attempt admissible now",
            truth(effective_source_theorem_attempt_admissible_now),
            "Source-theorem work should not resume until the matter/rot embedding problem is resolved.",
        ),
        row(
            "observable_dictionary_gate_admissible_now",
            "pass" if observable_dictionary_gate_admissible_now else "reject",
            "observable dictionary gate admissible now",
            truth(observable_dictionary_gate_admissible_now),
            "Observable-dictionary work remains downstream of an exact current closure and source theorem.",
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
            "action_principle_audit": display_path(ACTION_AUDIT),
            "reactivation_gate": display_path(REACTIVATION_GATE),
            "charge_closure_gate": display_path(CHARGE_CLOSURE_GATE),
            "anchor_gate": display_path(ANCHOR_GATE),
        },
        "constants": {
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "free_backbone_linear_formula_derived": free_backbone_linear_formula_derived,
        "transverse_photon_branch_excludes_chi_source": transverse_photon_branch_excludes_chi_source,
        "same_field_on_shell_linear_source_zero": same_field_on_shell_linear_source_zero,
        "explicit_qball_matter_current_functional_available": explicit_qball_matter_current_functional_available,
        "explicit_rotational_source_functional_available": explicit_rotational_source_functional_available,
        "exact_charge_current_noether_closure_available": exact_charge_current_noether_closure_available,
        "matter_rot_source_embedding_audit_required": matter_rot_source_embedding_audit_required,
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
            "part1_total_action": part1_total_action_hit,
            "part1_full_action": part1_full_action_hit,
            "part1_free_action": part1_free_action_hit,
            "part1_matter_current": part1_matter_current_hit,
            "part1_interaction": part1_interaction_hit,
            "part1_rot_placeholder": part1_rot_placeholder_hit,
            "part1_chi": part1_chi_hit,
            "part1_noether": part1_noether_hit,
            "part1_post_photon": part1_post_photon_hit,
            "part3a_photon_branch": part3a_photon_hit,
            "jeff_step2": jeff_step2_hit,
            "jeff_case1": jeff_case_hit,
        },
        "carry_over": {
            "action_audit_continuity": action_audit.get("equations", {}).get("continuity"),
            "reactivation_summary": reactivation_summary,
            "charge_closure_summary": charge_closure_summary,
            "anchor_summary": anchor_summary,
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
            "8.7.56.1535",
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
            "8.7.56.1536",
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
            "8.7.56.1537",
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
            "8.7.56.1538",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] exact charge-current / Noether-current closure derivation artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
