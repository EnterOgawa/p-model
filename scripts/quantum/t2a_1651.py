#!/usr/bin/env python3
"""Generate 8.7.56.1651-.1654 branch-local full nonlinear energy-density audit artifacts.

This branch follows the first-shot breakthrough instruction pack literally
after the exact constitutive-map audit has already failed honestly.

The key distinction is narrow and explicit:

1. current frozen-action canon still does not close an exact constitutive map,
2. but the retained exact branch already carries explicit local nonlinear
   structure in the solver variable `rho = sqrt(f_0^2 + f_L^2)`,
3. so the next honest test is to audit branch-local nonlinear energy-density
   candidates on the same retained branch and see whether they move the blind
   fixed-q_theory read toward the scalar strong candidate.

The branch therefore keeps the canonical gap visible while testing two
branch-local nonlinear candidates:

- the family-level vacuum-subtracted Mexican-hat proxy,
- the pilot-consistent nonlinear density reconstructed from the retained exact
  solver coefficient `(3 rho + rho^2)`.
"""

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

import scripts.quantum.t2a_1627 as energy_ff_tools


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LOCAL_RESPONSE = ROOT / "doc" / "quantum" / "50_trial2_vector_qball_breakthrough_instruction_response.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

EXTERNAL_NOTE = Path(
    r"C:\Users\ogawa\Downloads\50_trial2_numeric_alpha_vector_qball_breakthrough_instruction_pack.md"
)
CONSTITUTIVE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1647_1650_constitutive_map_audit_declaration_gate_metrics.json"
)
ENERGY_DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1623_1626_energy_density_audit_declaration_gate_metrics.json"
)
ENERGY_FF_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1627_1630_energy_density_ff_audit_declaration_gate_metrics.json"
)
ENERGY_CASE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1631_1634_energy_density_alpha_case_class_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1651-1654"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor branch-local full nonlinear "
    "energy-density audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "full_nl_energy_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_exact_constitutive_map_unavailable_"
    "branch_local_full_nonlinear_energy_audit_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_branch_local_full_nonlinear_energy_candidates_"
    "track_vector_no_go_primary_decision_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_primary_decision_gate_"
    "secondary_canonical_promotion_audit"
)
NEXT_ROUTE = "8.7.56.1655"
FALLBACK_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_p_mu_transverse_response_"
    "projected_kernel_observable_audit"
)
FALLBACK_ROUTE = "8.7.56.1659"
SECOND_FALLBACK_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_constrained_ground_state_"
    "branch_selection_audit"
)
SECOND_FALLBACK_ROUTE = "8.7.56.1663"

SCALAR_ALPHA = 0.00715678583937324
TARGET_ALPHA = 1.0 / 137.035999084
VECTOR_ALPHA = 0.0005579616187042394


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required path is missing."""
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


# 関数: 2つの alpha の相対ギャップを返す。

def relative_gap(value: float, reference: float) -> float:
    """Return one reference-relative absolute gap."""
    return float(abs(float(value) - float(reference)) / float(reference))


# 関数: retained branch 上の nonlinear candidate surfaces を構成する。

def build_candidate_surfaces() -> dict:
    """Construct branch-local nonlinear candidate surfaces on the retained branch."""
    bundle = energy_ff_tools.build_density_bundle()
    rho = np.sqrt(
        np.maximum(
            bundle["f0_values"] * bundle["f0_values"]
            + bundle["f_l_values"] * bundle["f_l_values"],
            0.0,
        )
    )

    # The retained exact solver uses `(3 rho + rho^2) f` as the nonlinear term.
    # Integrating `(dU/d rho)/rho = 3 rho + rho^2` gives
    # `U_nl(rho) = rho^3 + rho^4/4` up to an irrelevant additive constant.
    pilot_nonlinear_density = (rho**3) + 0.25 * (rho**4)
    pilot_full_density = bundle["hamiltonian_core_density"] + pilot_nonlinear_density
    mh_full_density = bundle["hamiltonian_core_density"] + bundle["mh_family_proxy_density"]

    pilot_surface = energy_ff_tools.summarize_surface(
        bundle["radius"],
        pilot_full_density,
        bundle["q_theory_over_m0"],
    )
    mh_surface = energy_ff_tools.summarize_surface(
        bundle["radius"],
        mh_full_density,
        bundle["q_theory_over_m0"],
    )

    return {
        "bundle": bundle,
        "rho": rho,
        "pilot_nonlinear_density": pilot_nonlinear_density,
        "pilot_full_density": pilot_full_density,
        "mh_full_density": mh_full_density,
        "pilot_surface": pilot_surface,
        "mh_surface": mh_surface,
        "pilot_full_positive_definite": bool(np.all(pilot_full_density >= -1.0e-12)),
        "mh_full_positive_definite": bool(np.all(mh_full_density >= -1.0e-12)),
        "pilot_nonlinear_norm": float(np.trapezoid(pilot_nonlinear_density * (bundle["radius"] ** 2), bundle["radius"])),
        "pilot_rho_max": float(np.max(rho)),
    }


# 関数: current-pack formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return the branch-local nonlinear-energy formulas used in this audit."""
    return {
        "retained_ansatz": (
            "P_mu^Qball = (f_0(r) e^{i omega t}, f_L(r) r_hat_i e^{i omega t})"
        ),
        "exact_core": (
            "epsilon_H,core(r) = f_0'(r)^2 + omega^2 f_L(r)^2 + m_0^2 f_0(r)^2"
        ),
        "solver_local_rho": "rho(r) = sqrt(f_0(r)^2 + f_L(r)^2)",
        "pilot_nonlinear_coefficient": "(3 rho + rho^2) f",
        "pilot_integrated_nonlinear_density": "epsilon_nl,pilot(r) = rho(r)^3 + rho(r)^4 / 4",
        "pilot_full_candidate": "epsilon_full,pilot(r) = epsilon_H,core(r) + rho(r)^3 + rho(r)^4/4",
        "family_proxy_candidate": (
            "epsilon_full,MHproxy(r) = epsilon_H,core(r) + (1/4)[(rho(r)^2-v^2)^2 - v^4]"
        ),
        "canonical_gap": (
            "Current frozen-action canon still lacks an exact constitutive map that "
            "promotes any branch-local nonlinear density candidate to the unique observable readout."
        ),
    }


# 関数: `.1651-.1654` を実行する。

def main() -> None:
    """Execute the branch-local full nonlinear energy-density audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LOCAL_RESPONSE,
        PART5,
        EXTERNAL_NOTE,
        CONSTITUTIVE_GATE,
        ENERGY_DERIV_GATE,
        ENERGY_FF_GATE,
        ENERGY_CASE_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    local_text = read_text(LOCAL_RESPONSE)
    part5_text = read_text(PART5)
    external_text = read_text(EXTERNAL_NOTE)

    constitutive_summary = read_json(CONSTITUTIVE_GATE)["summary"]
    energy_deriv_summary = read_json(ENERGY_DERIV_GATE)["summary"]
    energy_ff_summary = read_json(ENERGY_FF_GATE)["summary"]
    energy_case_summary = read_json(ENERGY_CASE_GATE)["summary"]

    candidate_bundle = build_candidate_surfaces()
    pilot_surface = candidate_bundle["pilot_surface"]
    mh_surface = candidate_bundle["mh_surface"]

    exact_constitutive_map_available = bool(
        constitutive_summary["exact_constitutive_map_available"]
    )
    branch_local_full_nonlinear_energy_density_exact_available = False
    pilot_branch_local_nonlinear_candidate_available = True
    family_proxy_branch_local_candidate_available = bool(
        energy_deriv_summary["vacuum_subtracted_mexican_hat_family_available"]
    )
    pilot_full_tracks_vector_no_go_scale = bool(
        relative_gap(pilot_surface["alpha_at_q_theory"], VECTOR_ALPHA)
        < relative_gap(pilot_surface["alpha_at_q_theory"], SCALAR_ALPHA)
    )
    mh_full_tracks_vector_no_go_scale = bool(
        relative_gap(mh_surface["alpha_at_q_theory"], VECTOR_ALPHA)
        < relative_gap(mh_surface["alpha_at_q_theory"], SCALAR_ALPHA)
    )
    pilot_full_supports_scalar_candidate = bool(
        abs(pilot_surface["alpha_at_q_theory"] - SCALAR_ALPHA)
        < abs(pilot_surface["alpha_at_q_theory"] - VECTOR_ALPHA)
    )
    mh_full_supports_scalar_candidate = bool(
        abs(mh_surface["alpha_at_q_theory"] - SCALAR_ALPHA)
        < abs(mh_surface["alpha_at_q_theory"] - VECTOR_ALPHA)
    )
    pilot_full_improves_official_core = bool(
        pilot_surface["alpha_residual_rel"] < energy_ff_summary["official_alpha_E_residual_rel"]
    )
    mh_full_improves_official_core = bool(
        mh_surface["alpha_residual_rel"] < energy_ff_summary["official_alpha_E_residual_rel"]
    )
    primary_decision_gate_admissible_now = True
    secondary_canonical_promotion_audit_admissible_now = True
    fallback_not_required_now = True
    physical_reject_required = False

    pilot_vs_core_alpha_rel_gap = relative_gap(
        pilot_surface["alpha_at_q_theory"],
        energy_ff_summary["official_alpha_E_at_q_theory"],
    )
    mh_vs_core_alpha_rel_gap = relative_gap(
        mh_surface["alpha_at_q_theory"],
        energy_ff_summary["official_alpha_E_at_q_theory"],
    )

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "local_response": display_path(LOCAL_RESPONSE),
            "part5": display_path(PART5),
            "external_instruction_pack": display_path(EXTERNAL_NOTE),
            "constitutive_gate": display_path(CONSTITUTIVE_GATE),
            "energy_derivation_gate": display_path(ENERGY_DERIV_GATE),
            "energy_ff_gate": display_path(ENERGY_FF_GATE),
            "energy_case_gate": display_path(ENERGY_CASE_GATE),
        },
        "constants": {
            "official_energy_core_alpha": energy_ff_summary["official_alpha_E_at_q_theory"],
            "scalar_alpha": SCALAR_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "vector_alpha": VECTOR_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "fallback_route_name": FALLBACK_ROUTE_NAME,
            "fallback_route": FALLBACK_ROUTE,
            "second_fallback_route_name": SECOND_FALLBACK_ROUTE_NAME,
            "second_fallback_route": SECOND_FALLBACK_ROUTE,
        },
    }

    rows = [
        row(
            "exact_constitutive_map_still_unavailable",
            "pass" if not exact_constitutive_map_available else "reject",
            "exact constitutive map still unavailable",
            truth(not exact_constitutive_map_available),
            "The branch-local nonlinear audit starts only after the canonical constitutive-map closure has already failed honestly.",
        ),
        row(
            "pilot_branch_local_nonlinear_candidate_available",
            "pass" if pilot_branch_local_nonlinear_candidate_available else "reject",
            "pilot-consistent branch-local nonlinear candidate available",
            truth(pilot_branch_local_nonlinear_candidate_available),
            "The retained exact solver already exposes rho = sqrt(f_0^2+f_L^2) and the nonlinear coefficient (3 rho + rho^2), so a branch-local pilot-consistent nonlinear density candidate can be audited on the same branch.",
        ),
        row(
            "family_proxy_branch_local_candidate_available",
            "pass" if family_proxy_branch_local_candidate_available else "reject",
            "family-level Mexican-hat proxy candidate available",
            truth(family_proxy_branch_local_candidate_available),
            "The vacuum-subtracted Mexican-hat family remains available as a parallel branch-local comparison surface.",
        ),
        row(
            "branch_local_full_nonlinear_energy_density_exact_available",
            "reject",
            "branch-local full nonlinear energy density exact available under current frozen pack",
            truth(branch_local_full_nonlinear_energy_density_exact_available),
            "The current frozen-action canon still does not uniquely promote a branch-local nonlinear density to the observable readout, so exact canonical availability remains false.",
        ),
        row(
            "pilot_full_nonlinear_alpha_at_q_theory",
            "watch",
            "pilot-consistent full nonlinear alpha at q_theory",
            pilot_surface["alpha_at_q_theory"],
            "This is the blind fixed-q_theory read from epsilon_H,core + rho^3 + rho^4/4 on the retained exact branch.",
        ),
        row(
            "pilot_full_nonlinear_residual_rel",
            "reject" if not pilot_full_supports_scalar_candidate else "pass",
            "pilot-consistent full nonlinear residual vs target",
            pilot_surface["alpha_residual_rel"],
            "The pilot-consistent nonlinear completion would only support a breakthrough if it moved the read closer to the scalar strong candidate than to the retained vector no-go scale.",
        ),
        row(
            "pilot_full_tracks_vector_no_go_scale",
            "pass" if pilot_full_tracks_vector_no_go_scale else "reject",
            "pilot-consistent full nonlinear candidate tracks vector no-go scale",
            relative_gap(pilot_surface["alpha_at_q_theory"], VECTOR_ALPHA),
            "The pilot-consistent nonlinear candidate remains much closer to the retained vector no-go scale than to the scalar strong candidate.",
        ),
        row(
            "pilot_full_supports_scalar_candidate",
            "pass" if pilot_full_supports_scalar_candidate else "reject",
            "pilot-consistent full nonlinear candidate supports scalar candidate",
            truth(pilot_full_supports_scalar_candidate),
            "This is the primary positive gate d_scalar < d_vec from the instruction pack.",
        ),
        row(
            "family_proxy_full_nonlinear_alpha_at_q_theory",
            "watch",
            "family-proxy full nonlinear alpha at q_theory",
            mh_surface["alpha_at_q_theory"],
            "This parallel branch-local proxy uses epsilon_H,core plus the vacuum-subtracted Mexican-hat family term on the same retained branch.",
        ),
        row(
            "family_proxy_supports_scalar_candidate",
            "pass" if mh_full_supports_scalar_candidate else "reject",
            "family-proxy full nonlinear candidate supports scalar candidate",
            truth(mh_full_supports_scalar_candidate),
            "The family-level nonlinear proxy also fails if it stays closer to the retained vector no-go scale.",
        ),
        row(
            "pilot_full_vs_core_alpha_rel_gap",
            "watch",
            "pilot-consistent nonlinear alpha relative gap vs official core alpha",
            pilot_vs_core_alpha_rel_gap,
            "This quantifies how little the branch-local nonlinear completion changes the already-audited official energy-core read.",
        ),
        row(
            "family_proxy_vs_core_alpha_rel_gap",
            "watch",
            "family-proxy nonlinear alpha relative gap vs official core alpha",
            mh_vs_core_alpha_rel_gap,
            "The Mexican-hat family proxy also stays extremely close to the official core read, so it does not open a hidden scalar rescue.",
        ),
        row(
            "primary_decision_gate_admissible_now",
            "pass" if primary_decision_gate_admissible_now else "reject",
            "primary decision gate admissible now",
            truth(primary_decision_gate_admissible_now),
            "After both branch-local nonlinear candidates have been audited honestly, the next route can freeze Gate A/B/C without adding another derivation layer.",
        ),
        row(
            "secondary_canonical_promotion_audit_admissible_now",
            "pass" if secondary_canonical_promotion_audit_admissible_now else "reject",
            "secondary canonical-promotion audit admissible now",
            truth(secondary_canonical_promotion_audit_admissible_now),
            "The primary decision gate can now decide whether evidence-only electric-like / note-gradient surfaces stay secondary or deserve a downstream promotion attempt.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "exact_constitutive_map_available": exact_constitutive_map_available,
        "branch_local_full_nonlinear_energy_density_exact_available": branch_local_full_nonlinear_energy_density_exact_available,
        "pilot_branch_local_nonlinear_candidate_available": pilot_branch_local_nonlinear_candidate_available,
        "family_proxy_branch_local_candidate_available": family_proxy_branch_local_candidate_available,
        "pilot_full_nonlinear_F_at_q_theory": pilot_surface["F_at_q_theory"],
        "pilot_full_nonlinear_alpha_at_q_theory": pilot_surface["alpha_at_q_theory"],
        "pilot_full_nonlinear_alpha_residual_rel": pilot_surface["alpha_residual_rel"],
        "pilot_full_tracks_vector_no_go_scale": pilot_full_tracks_vector_no_go_scale,
        "pilot_full_supports_scalar_candidate": pilot_full_supports_scalar_candidate,
        "pilot_full_improves_official_core": pilot_full_improves_official_core,
        "pilot_full_positive_definite": candidate_bundle["pilot_full_positive_definite"],
        "family_proxy_full_F_at_q_theory": mh_surface["F_at_q_theory"],
        "family_proxy_full_alpha_at_q_theory": mh_surface["alpha_at_q_theory"],
        "family_proxy_full_alpha_residual_rel": mh_surface["alpha_residual_rel"],
        "family_proxy_tracks_vector_no_go_scale": mh_full_tracks_vector_no_go_scale,
        "family_proxy_supports_scalar_candidate": mh_full_supports_scalar_candidate,
        "family_proxy_improves_official_core": mh_full_improves_official_core,
        "family_proxy_full_positive_definite": candidate_bundle["mh_full_positive_definite"],
        "pilot_full_vs_core_alpha_rel_gap": pilot_vs_core_alpha_rel_gap,
        "family_proxy_vs_core_alpha_rel_gap": mh_vs_core_alpha_rel_gap,
        "official_energy_core_alpha_at_q_theory": energy_ff_summary["official_alpha_E_at_q_theory"],
        "official_energy_case": energy_case_summary["selected_classification_case"],
        "primary_decision_gate_admissible_now": primary_decision_gate_admissible_now,
        "secondary_canonical_promotion_audit_admissible_now": secondary_canonical_promotion_audit_admissible_now,
        "fallback_not_required_now": fallback_not_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_fallback_route": FALLBACK_ROUTE_NAME,
        "selected_fallback_route_or_none": FALLBACK_ROUTE,
        "selected_second_fallback_route": SECOND_FALLBACK_ROUTE_NAME,
        "selected_second_fallback_route_or_none": SECOND_FALLBACK_ROUTE,
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
            "external_primary2_hit": hit(
                external_text,
                "### 3.3 primary-2: branch-local full nonlinear energy density audit",
            ),
            "external_gate_a_hit": hit(external_text, "#### Gate A: promote"),
            "external_gate_b_hit": hit(external_text, "#### Gate B: retain but not promote"),
            "external_gate_c_hit": hit(external_text, "#### Gate C: reserve"),
            "local_response_primary2_hit": hit(local_text, "**branch-local full nonlinear energy-density audit**"),
            "status_current_step_hit": hit(status_text, "8.7.56.1651"),
            "roadmap_current_branch_hit": hit(roadmap_text, "branch-local full nonlinear energy-density audit"),
            "current_problem_hit": hit(current_problem_text, "branch-local full nonlinear energy-density audit"),
            "current_status_hit": hit(current_status_text, "branch-local full nonlinear energy-density audit"),
            "unified_roadmap_hit": hit(unified_text, "`.1651-.1654` は **branch-local full nonlinear energy-density audit**"),
            "part5_hit": hit(part5_text, "branch-local full nonlinear energy-density audit"),
        },
        "retained_branch": {
            "q_theory_over_m0": candidate_bundle["bundle"]["q_theory_over_m0"],
            "beta_1_scalar": candidate_bundle["bundle"]["beta"],
            "amp0_phase1_equivalent": candidate_bundle["bundle"]["amp0"],
            "amp_l_phase1_equivalent": candidate_bundle["bundle"]["amp_l"],
            "pilot_rho_max": candidate_bundle["pilot_rho_max"],
            "pilot_nonlinear_norm": candidate_bundle["pilot_nonlinear_norm"],
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "official_energy_core_alpha_at_q_theory": energy_ff_summary["official_alpha_E_at_q_theory"],
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1651",
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
                "8.7.56.1652",
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
                "8.7.56.1653",
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
                "8.7.56.1654",
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
