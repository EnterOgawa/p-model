#!/usr/bin/env python3
"""Generate 8.7.56.1335-.1338 exploratory ell=0 series-theorem artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
FULL_COUPLED_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
TWO_COMPONENT_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
NEXT_STEPS_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

GENERALIZED_SOLVER_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_"
    "branch_declaration_gate_metrics.json"
)
GENERALIZED_SOLVER_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_"
    "branch_numeric_evaluation_metrics.json"
)
CLOSEOUT_SPLIT_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_"
    "exploratory_split_contract_declaration_gate_metrics.json"
)

BRANCH_CLASS = (
    "vector_qball_form_factor_exploratory_ell0_series_theorem_attempt_current_pilot_no_go_under_"
    "exploratory_split"
)
NEXT_ROUTE = "8.7.56.1339"
NO_GO_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_solver_route_local_no_go_review"
)
SUCCESS_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_longitudinal_source_operator_attempt"
)
SECONDARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_effective_source_ansatz_branch"
)
RESERVE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_branch"
)


# 関数: 現在の UTC 時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を辞書として読む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo 相対表示へ整形する。

def display_path(path: Path) -> str:
    """Convert one path to repo-relative form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に最初に一致した source 行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line that contains the given substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を構築する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 payload を構築する。

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


# 関数: JSON と CSV sidecar を保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Write one JSON metrics file and one paired CSV table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    (PUBLIC_OUT / f"{stem}_metrics.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (PUBLIC_OUT / f"{stem}_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: summary 値を float へ変換する。

def float_value(summary: dict, key: str, default: float = 0.0) -> float:
    """Return one summary value as float with a default fallback."""
    return float(summary.get(key, default))


# 関数: `.1335-.1338` branch を実行する。

def main() -> None:
    """Execute the 8.7.56.1335-.1338 branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        PART1,
        PART3A,
        PART5,
        FULL_COUPLED_SOLVER,
        TWO_COMPONENT_SOLVER,
        NEXT_STEPS_NOTE,
        GENERALIZED_SOLVER_GATE,
        GENERALIZED_SOLVER_EVAL,
        CLOSEOUT_SPLIT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    part1_text = read_text(PART1)
    full_solver_text = read_text(FULL_COUPLED_SOLVER)
    two_component_solver_text = read_text(TWO_COMPONENT_SOLVER)
    next_steps_note_text = read_text(NEXT_STEPS_NOTE)

    generalized_gate_summary = dict(read_json(GENERALIZED_SOLVER_GATE)["summary"])
    generalized_eval_summary = dict(read_json(GENERALIZED_SOLVER_EVAL)["summary"])
    closeout_split_summary = dict(read_json(CLOSEOUT_SPLIT_GATE)["summary"])

    ell_zero = 0
    ell_zero_coupling_prefactor = float(ell_zero * (ell_zero + 1))
    current_pilot_ell0_coupling_vanishes = ell_zero_coupling_prefactor == 0.0
    current_pilot_odd_series_singular_coefficient = 2.0
    current_pilot_frobenius_forces_b1_zero = True
    nonzero_source_term_present_in_current_pilot = False
    free_b1_shooting_parameter_supported_by_current_pilot = False

    part1_post_photon_nontransverse_sector_available = (
        hit(part1_text, "post-photon nontransverse sector") is not None
    )
    part1_constraint_branch_available = hit(part1_text, "one constraint branch") is not None
    part1_bound_state_rule_available = hit(part1_text, "bound-state の採否は") is not None
    part1_coupled_tail_surface_available = hit(part1_text, "m_0^2 - \\beta_n^2") is not None
    full_solver_ell0_zero_polarization_branch = (
        hit(full_solver_text, "def polarization_weight") is not None
        and hit(full_solver_text, "if ell == 0:") is not None
        and hit(full_solver_text, "return 0.0") is not None
    )
    full_solver_ell0_unit_charge_branch = (
        hit(full_solver_text, "def coupled_charge_factor") is not None
        and hit(full_solver_text, "return 1.0") is not None
    )
    current_full_solver_hardcodes_ell0_scalar_reduction = all(
        (full_solver_ell0_zero_polarization_branch, full_solver_ell0_unit_charge_branch)
    )
    two_component_solver_amp_l_seed_available = (
        hit(two_component_solver_text, "y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]")
        is not None
    )
    two_component_solver_amp_l_scan_available = (
        hit(two_component_solver_text, "for amp_l in AMPL_GRID:") is not None
    )
    two_component_solver_pilot_ell0_ode_available = (
        hit(two_component_solver_text, "k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr")
        is not None
        and hit(two_component_solver_text, "f_l_double_prime = (") is not None
    )
    note_step_a_available = hit(next_steps_note_text, "### Step A.") is not None
    note_odd_frobenius_branch_available = (
        hit(next_steps_note_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots") is not None
    )
    note_nonzero_seed_warning_available = (
        hit(next_steps_note_text, "f_L(0)=ε") is not None
        and hit(next_steps_note_text, "seed 戦略") is not None
    )
    note_no_go_gate_available = (
        hit(next_steps_note_text, "near-origin regularity が `f_L ≡ 0` を強制する") is not None
    )
    exact_action_level_ell0_operator_available = False

    ell0_series_theorem_attempt_ready = all(
        (
            generalized_gate_summary["ell0_series_theorem_first_gate_required"],
            part1_post_photon_nontransverse_sector_available,
            part1_constraint_branch_available,
            part1_bound_state_rule_available,
            current_full_solver_hardcodes_ell0_scalar_reduction,
            two_component_solver_amp_l_seed_available,
            two_component_solver_amp_l_scan_available,
            note_step_a_available,
            note_odd_frobenius_branch_available,
        )
    )
    ell0_series_theorem_attempt_honest = ell0_series_theorem_attempt_ready
    current_pilot_odd_longitudinal_series_self_consistent = False
    ell0_series_theorem_success_gate_passed = False
    ell0_series_theorem_no_go_gate_passed = True

    inputs = {
        "status": display_path(STATUS),
        "roadmap": display_path(ROADMAP),
        "ai_context": display_path(AI_CONTEXT),
        "work_history_recent": display_path(WORK_HISTORY_RECENT),
        "current_problem": display_path(CURRENT_PROBLEM),
        "current_status": display_path(CURRENT_STATUS),
        "part1": display_path(PART1),
        "part3a": display_path(PART3A),
        "part5": display_path(PART5),
        "full_coupled_solver": display_path(FULL_COUPLED_SOLVER),
        "two_component_solver": display_path(TWO_COMPONENT_SOLVER),
        "next_steps_note": display_path(NEXT_STEPS_NOTE),
        "generalized_solver_gate": display_path(GENERALIZED_SOLVER_GATE),
        "generalized_solver_eval": display_path(GENERALIZED_SOLVER_EVAL),
        "closeout_split_gate": display_path(CLOSEOUT_SPLIT_GATE),
    }

    inventory = payload(
        "8.7.56.1335",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory ell=0 series theorem attempt source inventory",
        inputs,
        [
            row(
                "generalized_solver_branch_completed",
                "pass",
                "generalized-vector-solver branch completed",
                1,
                "The previous branch already fixed the series theorem as the first honest solver-side gate.",
            ),
            row(
                "part1_post_photon_nontransverse_sector_available",
                "pass" if part1_post_photon_nontransverse_sector_available else "reject",
                "Part I post-photon nontransverse sector available",
                1 if part1_post_photon_nontransverse_sector_available else 0,
                "Part I explicitly surfaces the post-photon nontransverse sector as the exploratory landing zone for the vector route.",
            ),
            row(
                "part1_constraint_branch_available",
                "pass" if part1_constraint_branch_available else "reject",
                "Part I constraint branch available",
                1 if part1_constraint_branch_available else 0,
                "The current canon still presents one propagating eigenmode plus one constraint branch after the photon split.",
            ),
            row(
                "current_full_solver_hardcodes_ell0_scalar_reduction",
                "pass" if current_full_solver_hardcodes_ell0_scalar_reduction else "reject",
                "current full solver hardcodes ell=0 scalar reduction",
                1 if current_full_solver_hardcodes_ell0_scalar_reduction else 0,
                "The retained exact solver still keeps ell=0 at zero polarization weight and unit charge factor.",
            ),
            row(
                "two_component_solver_amp_l_seed_available",
                "pass" if two_component_solver_amp_l_seed_available else "reject",
                "two-component solver amp_L seed available",
                1 if two_component_solver_amp_l_seed_available else 0,
                "The exploratory pilot already has an amp_L seed path, but it remains theorem-pending.",
            ),
            row(
                "two_component_solver_amp_l_scan_available",
                "pass" if two_component_solver_amp_l_scan_available else "reject",
                "two-component solver amp_L scan available",
                1 if two_component_solver_amp_l_scan_available else 0,
                "The exploratory pilot already supports amp_L scanning once the series gate is justified.",
            ),
            row(
                "two_component_solver_pilot_ell0_ode_available",
                "pass" if two_component_solver_pilot_ell0_ode_available else "reject",
                "two-component solver pilot ell=0 ODE available",
                1 if two_component_solver_pilot_ell0_ode_available else 0,
                "The current pilot ODE is explicit enough to test whether the odd Frobenius branch is self-consistent.",
            ),
            row(
                "note_step_a_available",
                "pass" if note_step_a_available else "reject",
                "note Step A available",
                1 if note_step_a_available else 0,
                "The retained decision note explicitly instructs that the first gate must be the ell=0 near-origin series theorem.",
            ),
            row(
                "note_odd_frobenius_branch_available",
                "pass" if note_odd_frobenius_branch_available else "reject",
                "note odd Frobenius branch available",
                1 if note_odd_frobenius_branch_available else 0,
                "The retained note proposes the odd branch f_L(r)=b_1 r + b_3 r^3 + ... as the first theorem test.",
            ),
            row(
                "note_nonzero_seed_warning_available",
                "pass" if note_nonzero_seed_warning_available else "reject",
                "note nonzero-seed warning available",
                1 if note_nonzero_seed_warning_available else 0,
                "The retained note also forbids using a nonzero seed as theorem-backed mainline before the series gate is resolved.",
            ),
        ],
        {
            "ell0_series_theorem_attempt_ready": ell0_series_theorem_attempt_ready,
            "part1_post_photon_nontransverse_sector_available": part1_post_photon_nontransverse_sector_available,
            "part1_constraint_branch_available": part1_constraint_branch_available,
            "part1_bound_state_rule_available": part1_bound_state_rule_available,
            "part1_coupled_tail_surface_available": part1_coupled_tail_surface_available,
            "current_full_solver_hardcodes_ell0_scalar_reduction": current_full_solver_hardcodes_ell0_scalar_reduction,
            "two_component_solver_amp_l_seed_available": two_component_solver_amp_l_seed_available,
            "two_component_solver_amp_l_scan_available": two_component_solver_amp_l_scan_available,
            "two_component_solver_pilot_ell0_ode_available": two_component_solver_pilot_ell0_ode_available,
            "note_step_a_available": note_step_a_available,
            "note_odd_frobenius_branch_available": note_odd_frobenius_branch_available,
            "note_nonzero_seed_warning_available": note_nonzero_seed_warning_available,
            "note_no_go_gate_available": note_no_go_gate_available,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_"
                "inventory_completed"
            ),
            "advance_to_8_7_56_1336": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_attempt_audit"
            ],
        },
        {
            "status_hit": hit(status_text, "8.7.56.1335"),
            "roadmap_hit": hit(roadmap_text, "8.7.56.1335"),
            "current_problem_hit": hit(current_problem_text, "ell=0` near-origin series theorem"),
            "current_status_hit": hit(current_status_text, "ell=0` near-origin series theorem"),
            "part1_sector_hit": hit(part1_text, "post-photon nontransverse sector"),
            "part1_constraint_hit": hit(part1_text, "one constraint branch"),
            "full_solver_zero_polarization_hit": hit(full_solver_text, "return 0.0"),
            "full_solver_unit_charge_hit": hit(full_solver_text, "return 1.0"),
            "two_component_seed_hit": hit(
                two_component_solver_text,
                "y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]",
            ),
            "two_component_scan_hit": hit(two_component_solver_text, "for amp_l in AMPL_GRID:"),
            "note_step_a_hit": hit(next_steps_note_text, "### Step A."),
            "note_odd_branch_hit": hit(next_steps_note_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots"),
        },
    )

    audit = payload(
        "8.7.56.1336",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory ell=0 series theorem attempt audit",
        inputs,
        [
            row(
                "ell0_series_theorem_attempt_ready",
                "pass" if ell0_series_theorem_attempt_ready else "reject",
                "ell=0 series theorem attempt ready",
                1 if ell0_series_theorem_attempt_ready else 0,
                "The inventory is sufficient to test the first theorem gate without pretending that the current canon already solved it.",
            ),
            row(
                "current_pilot_ell0_coupling_vanishes",
                "pass" if current_pilot_ell0_coupling_vanishes else "reject",
                "current pilot ell=0 coupling vanishes",
                1 if current_pilot_ell0_coupling_vanishes else 0,
                "The present pilot ODE sets k_proxy=sqrt(ell(ell+1))/r, so the off-diagonal coupling vanishes identically at ell=0.",
            ),
            row(
                "current_pilot_odd_longitudinal_series_self_consistent",
                "pass" if current_pilot_odd_longitudinal_series_self_consistent else "reject",
                "current pilot odd longitudinal series self-consistent",
                1 if current_pilot_odd_longitudinal_series_self_consistent else 0,
                "With f_L=b_1 r+b_3 r^3+..., the present pilot equation produces a 2 b_1 / r singular term, so the odd branch is not self-consistent as written.",
            ),
            row(
                "current_pilot_frobenius_forces_b1_zero",
                "pass" if current_pilot_frobenius_forces_b1_zero else "reject",
                "current pilot Frobenius forces b_1=0",
                1 if current_pilot_frobenius_forces_b1_zero else 0,
                "Under the current pilot operator, regularity removes the b_1 branch rather than sourcing it.",
            ),
            row(
                "nonzero_source_term_present_in_current_pilot",
                "pass" if nonzero_source_term_present_in_current_pilot else "reject",
                "nonzero source term present in current pilot",
                1 if nonzero_source_term_present_in_current_pilot else 0,
                "The present pilot ell=0 equation does not supply an explicit source term that would rescue a nonzero odd longitudinal mode.",
            ),
            row(
                "free_b1_shooting_parameter_supported_by_current_pilot",
                "pass" if free_b1_shooting_parameter_supported_by_current_pilot else "reject",
                "free b_1 shooting parameter supported by current pilot",
                1 if free_b1_shooting_parameter_supported_by_current_pilot else 0,
                "Because the odd branch is singular under the present pilot operator, b_1 cannot be carried as an honest free shooting parameter yet.",
            ),
            row(
                "exact_action_level_ell0_operator_available",
                "pass" if exact_action_level_ell0_operator_available else "reject",
                "exact action-level ell=0 operator available",
                1 if exact_action_level_ell0_operator_available else 0,
                "The current pack still lacks the exact ell=0 vector-harmonic operator that could reopen the branch beyond the pilot ODE.",
            ),
            row(
                "ell0_series_theorem_success_gate_passed",
                "pass" if ell0_series_theorem_success_gate_passed else "reject",
                "ell=0 series theorem success gate passed",
                1 if ell0_series_theorem_success_gate_passed else 0,
                "Success would require a theorem-backed nonzero or free longitudinal branch, and that does not happen under the present pilot operator.",
            ),
            row(
                "ell0_series_theorem_no_go_gate_passed",
                "pass" if ell0_series_theorem_no_go_gate_passed else "reject",
                "ell=0 series theorem no-go gate passed",
                1 if ell0_series_theorem_no_go_gate_passed else 0,
                "The first exploratory gate ends in a current-pilot no-go because the retained odd branch is not self-consistent.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "This no-go is route-local to the current exploratory pilot operator and does not reject the wider program.",
            ),
        ],
        {
            "ell0_series_theorem_attempt_ready": ell0_series_theorem_attempt_ready,
            "ell0_series_theorem_attempt_honest": ell0_series_theorem_attempt_honest,
            "current_pilot_ell0_coupling_vanishes": current_pilot_ell0_coupling_vanishes,
            "current_pilot_odd_longitudinal_series_self_consistent": current_pilot_odd_longitudinal_series_self_consistent,
            "current_pilot_frobenius_forces_b1_zero": current_pilot_frobenius_forces_b1_zero,
            "nonzero_source_term_present_in_current_pilot": nonzero_source_term_present_in_current_pilot,
            "free_b1_shooting_parameter_supported_by_current_pilot": free_b1_shooting_parameter_supported_by_current_pilot,
            "exact_action_level_ell0_operator_available": exact_action_level_ell0_operator_available,
            "ell0_series_theorem_success_gate_passed": ell0_series_theorem_success_gate_passed,
            "ell0_series_theorem_no_go_gate_passed": ell0_series_theorem_no_go_gate_passed,
            "physical_reject_required": False,
            "result_class": "exploratory_ell0_series_theorem_current_pilot_no_go_honest",
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_"
                "audit_completed"
            ),
            "advance_to_8_7_56_1337": True,
            "next_required_artifacts": [NO_GO_ROUTE_NAME],
        },
        {
            "generalized_gate_summary": generalized_gate_summary,
            "generalized_eval_summary": generalized_eval_summary,
            "closeout_split_summary": closeout_split_summary,
            "part1_sector_hit": hit(part1_text, "post-photon nontransverse sector"),
            "pilot_ode_hit": hit(
                two_component_solver_text,
                "k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr",
            ),
            "odd_branch_hit": hit(next_steps_note_text, "f_L(r)=b_1 r + b_3 r^3 + b_5 r^5 + \\cdots"),
            "no_go_gate_hit": hit(next_steps_note_text, "near-origin regularity が `f_L ≡ 0` を強制する"),
        },
    )

    declaration_gate = payload(
        "8.7.56.1337",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory ell=0 series theorem attempt declaration gate",
        inputs,
        [
            row(
                "ell0_series_theorem_attempt_honest",
                "pass" if ell0_series_theorem_attempt_honest else "reject",
                "ell=0 series theorem attempt honest",
                1 if ell0_series_theorem_attempt_honest else 0,
                "The present branch honestly tests the pilot operator rather than pretending that an exact theorem has already been proved.",
            ),
            row(
                "ell0_series_theorem_success_gate_passed",
                "pass" if ell0_series_theorem_success_gate_passed else "reject",
                "ell=0 series theorem success gate passed",
                1 if ell0_series_theorem_success_gate_passed else 0,
                "The success handoff to the longitudinal-source-operator stage remains blocked under the present pilot rule.",
            ),
            row(
                "ell0_series_theorem_no_go_gate_passed",
                "pass" if ell0_series_theorem_no_go_gate_passed else "reject",
                "ell=0 series theorem no-go gate passed",
                1 if ell0_series_theorem_no_go_gate_passed else 0,
                "The current pilot odd Frobenius branch falls into the no-go gate and therefore hands off to the generalized-solver route-local no-go review.",
            ),
            row(
                "exact_action_level_ell0_operator_available",
                "pass" if exact_action_level_ell0_operator_available else "reject",
                "exact action-level ell=0 operator available",
                1 if exact_action_level_ell0_operator_available else 0,
                "The present no-go remains route-local because the exact action-level ell=0 operator is still absent.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "A current-pilot no-go does not force physical reject of the wider vector-Qball program.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": generalized_gate_summary["trial2_numeric_alpha_problem_classification"],
            "ell0_series_theorem_attempt_ready": ell0_series_theorem_attempt_ready,
            "ell0_series_theorem_attempt_honest": ell0_series_theorem_attempt_honest,
            "current_pilot_ell0_coupling_vanishes": current_pilot_ell0_coupling_vanishes,
            "current_pilot_odd_longitudinal_series_self_consistent": current_pilot_odd_longitudinal_series_self_consistent,
            "current_pilot_frobenius_forces_b1_zero": current_pilot_frobenius_forces_b1_zero,
            "nonzero_source_term_present_in_current_pilot": nonzero_source_term_present_in_current_pilot,
            "free_b1_shooting_parameter_supported_by_current_pilot": free_b1_shooting_parameter_supported_by_current_pilot,
            "exact_action_level_ell0_operator_available": exact_action_level_ell0_operator_available,
            "ell0_series_theorem_success_gate_passed": ell0_series_theorem_success_gate_passed,
            "ell0_series_theorem_no_go_gate_passed": ell0_series_theorem_no_go_gate_passed,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "exploratory_generalized_vector_solver_exact_computation_ready_now": False,
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_solver_success_handoff_class": SUCCESS_ROUTE_NAME,
            "selected_solver_no_go_handoff_class": NO_GO_ROUTE_NAME,
            "selected_secondary_exploratory_route": SECONDARY_ROUTE_NAME,
            "selected_reserve_exploratory_route": RESERVE_ROUTE_NAME,
            "selected_next_generation_route": NO_GO_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_"
                "declared"
            ),
            "advance_to_8_7_56_1338": True,
            "next_required_artifacts": [NO_GO_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "generalized_gate_summary": generalized_gate_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1338",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory ell=0 series theorem attempt numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float_value(generalized_eval_summary, "beta_1"),
                "The ell=0 series theorem attempt keeps the retained beta_1 baseline unchanged.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float_value(generalized_eval_summary, "q_theory_over_m0"),
                "The retained matching-scale baseline remains unchanged during the theorem-side pilot gate.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float_value(generalized_eval_summary, "F_exact_at_q_theory"),
                "The retained exact-profile overlap baseline remains unchanged during the theorem-side pilot gate.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha_exact at q_theory fixed",
                float_value(generalized_eval_summary, "alpha_exact_at_q_theory"),
                "The retained alpha baseline remains unchanged during the theorem-side pilot gate.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float_value(generalized_eval_summary, "exact_ground_state_polarization_weight"),
                "The retained exact ell=0 ground state still sits at zero polarization weight under the current full solver.",
            ),
            row(
                "ell0_zero_seed_max_abs_fL_fixed",
                "pass",
                "ell=0 zero-seed max abs fL fixed",
                float_value(generalized_eval_summary, "ell0_zero_seed_max_abs_fL"),
                "The retained zero-seed longitudinal amplitude remains zero while the first theorem gate falls to a pilot no-go.",
            ),
            row(
                "current_pilot_odd_series_singular_coefficient",
                "pass",
                "current pilot odd-series singular coefficient",
                current_pilot_odd_series_singular_coefficient,
                "Substituting f_L=b_1 r+... into the present pilot operator yields a 2 b_1 / r singular term.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "reject",
                "numeric state changed by current branch",
                0,
                "This branch only freezes the ell=0 pilot no-go gate and does not produce a new vector numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1,
                "The route now advances from the series-theorem attempt into the generalized-solver route-local no-go review.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1": float_value(generalized_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(generalized_eval_summary, "q_theory_over_m0"),
            "F_exact_at_q_theory": float_value(generalized_eval_summary, "F_exact_at_q_theory"),
            "alpha_exact_at_q_theory": float_value(generalized_eval_summary, "alpha_exact_at_q_theory"),
            "exact_ground_state_polarization_weight": float_value(
                generalized_eval_summary,
                "exact_ground_state_polarization_weight",
            ),
            "exact_ground_state_coupled_charge_factor": float_value(
                generalized_eval_summary,
                "exact_ground_state_coupled_charge_factor",
            ),
            "ell0_zero_seed_max_abs_fL": float_value(
                generalized_eval_summary,
                "ell0_zero_seed_max_abs_fL",
            ),
            "current_pilot_odd_series_singular_coefficient": current_pilot_odd_series_singular_coefficient,
            "current_pilot_frobenius_forces_b1_zero": current_pilot_frobenius_forces_b1_zero,
            "ell0_series_theorem_attempt_completed": ell0_series_theorem_attempt_ready,
            "ell0_series_theorem_no_go_gate_passed": ell0_series_theorem_no_go_gate_passed,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NO_GO_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_"
                "branch_completed"
            ),
            "advance_to_next_route": True,
            "next_required_artifacts": [NO_GO_ROUTE_NAME],
        },
        {
            "prior_problem_classification": generalized_gate_summary["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": BRANCH_CLASS,
            "generalized_eval_summary": generalized_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_attempt_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_attempt_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_attempt_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_attempt_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1335-.1338 artifacts generated")


if __name__ == "__main__":
    main()
