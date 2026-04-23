#!/usr/bin/env python3
"""Generate 8.7.56.1331-.1334 exploratory generalized-vector-solver artifacts."""

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

CLOSEOUT_SPLIT_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_"
    "exploratory_split_contract_declaration_gate_metrics.json"
)
CLOSEOUT_SPLIT_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_"
    "exploratory_split_contract_numeric_evaluation_metrics.json"
)
ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "declaration_gate_metrics.json"
)
CLOSURE_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_"
    "closure_gap_contract_declaration_gate_metrics.json"
)

BRANCH_CLASS = (
    "vector_qball_form_factor_exploratory_generalized_vector_solver_branch_under_exploratory_split"
)
NEXT_ROUTE = "8.7.56.1335"
ELL0_SERIES_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_attempt"
)
SUCCESS_HANDOFF_CLASS = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_longitudinal_source_operator_attempt"
)
NO_GO_HANDOFF_CLASS = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_solver_route_local_no_go_review"
)
SECONDARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_effective_source_ansatz_branch"
)
RESERVE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_branch"
)


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort when one required input is missing.

def require(path: Path) -> None:
    """Abort when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: convert one path to repo-relative display form when possible.

def display_path(path: Path) -> str:
    """Convert one path to repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching line for one substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build one standard payload.

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


# Function: write one JSON metrics payload and CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one JSON metrics payload and CSV rows table."""
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


# Function: convert one summary value to float with default fallback.

def float_value(summary: dict, key: str, default: float = 0.0) -> float:
    """Convert one summary value to float with default fallback."""
    return float(summary.get(key, default))


# Function: execute the 8.7.56.1331-.1334 branch.

def main() -> None:
    """Execute the 8.7.56.1331-.1334 branch."""
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
        CLOSEOUT_SPLIT_GATE,
        CLOSEOUT_SPLIT_EVAL,
        ROUTE_LOCAL_GATE,
        CLOSURE_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    full_solver_text = read_text(FULL_COUPLED_SOLVER)
    two_component_solver_text = read_text(TWO_COMPONENT_SOLVER)
    next_steps_note_text = read_text(NEXT_STEPS_NOTE)

    closeout_split_gate_summary = dict(read_json(CLOSEOUT_SPLIT_GATE)["summary"])
    closeout_split_eval_summary = dict(read_json(CLOSEOUT_SPLIT_EVAL)["summary"])
    route_local_gate_summary = dict(read_json(ROUTE_LOCAL_GATE)["summary"])
    closure_gate_summary = dict(read_json(CLOSURE_GATE)["summary"])

    current_canon_closeout_completed = bool(
        closeout_split_gate_summary["current_canon_closeout_completed"]
    )
    exploratory_split_completed = bool(closeout_split_eval_summary["exploratory_split_completed"])
    generalized_vector_solver_branch_candidate = bool(
        closeout_split_gate_summary["generalized_vector_solver_branch_admissible"]
    )
    effective_source_ansatz_branch_candidate = bool(
        closeout_split_gate_summary["effective_source_ansatz_branch_admissible"]
    )
    observable_dictionary_branch_candidate = bool(
        closeout_split_gate_summary["observable_dictionary_branch_admissible"]
    )
    route_local_no_go_theorem_retained = bool(
        route_local_gate_summary["route_local_no_go_theorem_honest"]
    )
    closure_gap_retained = not bool(
        closure_gate_summary["ground_state_two_component_closure_already_implied_under_current_pack"]
    )

    part1_post_photon_nontransverse_sector_available = (
        hit(part1_text, "post-photon nontransverse sector") is not None
    )
    part1_constraint_branch_available = hit(part1_text, "one constraint branch") is not None
    part1_bound_state_localization_rule_available = (
        hit(part1_text, "bound-state の採否は") is not None
    )
    full_solver_ell0_zero_polarization_branch = (
        hit(full_solver_text, "if ell == 0:") is not None
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
    two_component_solver_supports_amp_l_exploration = all(
        (two_component_solver_amp_l_seed_available, two_component_solver_amp_l_scan_available)
    )
    next_steps_series_first_program_available = (
        hit(next_steps_note_text, "Step A. `ℓ=0` の exact ansatz と near-origin series を固定する")
        is not None
    )
    next_steps_nonzero_seed_not_first = (
        hit(next_steps_note_text, "次にやるべきは nonzero seed を入れることそのものではなく")
        is not None
    )
    next_steps_success_gate_available = (
        hit(next_steps_note_text, "near-origin series が nonzero `f_L` を許す") is not None
    )
    next_steps_no_go_gate_available = (
        hit(next_steps_note_text, "near-origin regularity が `f_L ≡ 0` を強制する") is not None
    )

    current_canon_already_proves_nonzero_ell0_temporal_longitudinal_closure = False
    nonzero_seed_mainline_admissible_without_series = False
    ell0_series_theorem_first_gate_required = all(
        (
            current_canon_closeout_completed,
            exploratory_split_completed,
            generalized_vector_solver_branch_candidate,
            route_local_no_go_theorem_retained,
            closure_gap_retained,
            part1_post_photon_nontransverse_sector_available,
            part1_constraint_branch_available,
            current_full_solver_hardcodes_ell0_scalar_reduction,
            two_component_solver_supports_amp_l_exploration,
            next_steps_series_first_program_available,
            next_steps_nonzero_seed_not_first,
        )
    )
    solver_boundary_value_reformulation_admissible = all(
        (
            ell0_series_theorem_first_gate_required,
            part1_bound_state_localization_rule_available,
            next_steps_success_gate_available,
            next_steps_no_go_gate_available,
        )
    )
    exploratory_generalized_vector_solver_branch_ready = all(
        (
            current_canon_closeout_completed,
            generalized_vector_solver_branch_candidate,
            closure_gap_retained,
            route_local_no_go_theorem_retained,
            ell0_series_theorem_first_gate_required,
            solver_boundary_value_reformulation_admissible,
        )
    )
    exploratory_generalized_vector_solver_branch_honest = all(
        (
            exploratory_generalized_vector_solver_branch_ready,
            not current_canon_already_proves_nonzero_ell0_temporal_longitudinal_closure,
            not nonzero_seed_mainline_admissible_without_series,
        )
    )
    exploratory_generalized_vector_solver_exact_computation_ready_now = False
    solver_success_gate_ready = all(
        (
            exploratory_generalized_vector_solver_branch_ready,
            next_steps_success_gate_available,
        )
    )
    solver_no_go_gate_ready = all(
        (
            exploratory_generalized_vector_solver_branch_ready,
            next_steps_no_go_gate_available,
        )
    )

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "full_coupled_solver": display_path(FULL_COUPLED_SOLVER),
            "two_component_solver": display_path(TWO_COMPONENT_SOLVER),
            "next_steps_note": display_path(NEXT_STEPS_NOTE),
        },
        "prior_metrics": {
            "closeout_split_gate": display_path(CLOSEOUT_SPLIT_GATE),
            "closeout_split_eval": display_path(CLOSEOUT_SPLIT_EVAL),
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
            "closure_gate": display_path(CLOSURE_GATE),
        },
        "constants": {
            "beta_1": float_value(closeout_split_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(closeout_split_eval_summary, "q_theory_over_m0"),
            "selected_first_gate_route": ELL0_SERIES_ROUTE_NAME,
            "selected_success_handoff_class": SUCCESS_HANDOFF_CLASS,
            "selected_no_go_handoff_class": NO_GO_HANDOFF_CLASS,
            "secondary_route_name": SECONDARY_ROUTE_NAME,
            "reserve_route_name": RESERVE_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1331",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory generalized-vector-solver branch source inventory",
        inputs,
        [
            row(
                "exploratory_generalized_vector_solver_inventory_ready",
                "pass" if exploratory_generalized_vector_solver_branch_ready else "reject",
                "exploratory generalized-vector-solver inventory ready",
                1 if exploratory_generalized_vector_solver_branch_ready else 0,
                "The exploratory solver inventory is ready only after current-canon closeout, closure-gap retain, route-local no-go retain, solver evidence, and the next-steps theorem program all align.",
            ),
            row(
                "part1_post_photon_nontransverse_sector_available",
                "pass" if part1_post_photon_nontransverse_sector_available else "reject",
                "Part I post-photon nontransverse sector available",
                1 if part1_post_photon_nontransverse_sector_available else 0,
                "The generalized solver branch needs the generic post-photon nontransverse sector wording before any exploratory closure work is admissible.",
            ),
            row(
                "current_full_solver_hardcodes_ell0_scalar_reduction",
                "pass" if current_full_solver_hardcodes_ell0_scalar_reduction else "reject",
                "current full solver hardcodes ell=0 scalar reduction",
                1 if current_full_solver_hardcodes_ell0_scalar_reduction else 0,
                "The current full solver still sets the ell=0 polarization weight to zero and keeps the coupled charge factor at unity, so it cannot by itself answer the ell=0 closure question.",
            ),
            row(
                "two_component_solver_supports_amp_l_exploration",
                "pass" if two_component_solver_supports_amp_l_exploration else "reject",
                "two-component solver supports amp_l exploration",
                1 if two_component_solver_supports_amp_l_exploration else 0,
                "The Trial-3 two-component pilot already has an explicit amp_l seed and scan loop, which makes a generalized solver branch operationally admissible.",
            ),
            row(
                "next_steps_series_first_program_available",
                "pass" if next_steps_series_first_program_available else "reject",
                "next-steps series-first program available",
                1 if next_steps_series_first_program_available else 0,
                "The retained expert note already says the first gate is ell=0 near-origin series, not a blind nonzero-seed push.",
            ),
            row(
                "current_canon_already_proves_nonzero_ell0_temporal_longitudinal_closure",
                "reject",
                "current canon already proves nonzero ell=0 temporal / longitudinal closure",
                0,
                "The exploratory branch starts precisely because the current pack does not already prove nonzero ell=0 closure.",
            ),
        ],
        {
            "exploratory_generalized_vector_solver_inventory_ready": exploratory_generalized_vector_solver_branch_ready,
            "current_canon_closeout_completed": current_canon_closeout_completed,
            "closure_gap_retained": closure_gap_retained,
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "part1_post_photon_nontransverse_sector_available": part1_post_photon_nontransverse_sector_available,
            "current_full_solver_hardcodes_ell0_scalar_reduction": current_full_solver_hardcodes_ell0_scalar_reduction,
            "two_component_solver_supports_amp_l_exploration": two_component_solver_supports_amp_l_exploration,
            "next_steps_series_first_program_available": next_steps_series_first_program_available,
            "selected_next_substep": "8.7.56.1332",
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_"
                "inventory_fixed"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_"
                "inventory_fixed"
            ),
            "advance_to_8_7_56_1332": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_branch_audit"
            ],
        },
        {
            "status_hits": {
                "status_1331": hit(status_text, "8.7.56.1331"),
                "status_exploratory_branch": hit(status_text, "exploratory generalized-vector-solver branch"),
            },
            "roadmap_hits": {
                "roadmap_1331": hit(roadmap_text, "`8.7.56.1331`"),
                "roadmap_1332": hit(roadmap_text, "`8.7.56.1332`"),
                "roadmap_1333": hit(roadmap_text, "`8.7.56.1333`"),
                "roadmap_1334": hit(roadmap_text, "`8.7.56.1334`"),
            },
            "problem_hits": {
                "problem_1331": hit(current_problem_text, "8.7.56.1331-.1334"),
                "problem_closure_gap": hit(current_problem_text, "source / closure / observable dictionary"),
            },
            "status_note_hits": {
                "status_1331": hit(current_status_text, "8.7.56.1331-.1334"),
                "status_generalized_solver": hit(current_status_text, "generalized-vector-solver branch"),
            },
            "paper_hits": {
                "part1_nontransverse": hit(part1_text, "post-photon nontransverse sector"),
                "part1_constraint": hit(part1_text, "one constraint branch"),
                "part3a_next": hit(part3a_text, "exploratory-generalized-vector-solver-branch next"),
                "part5_next": hit(part5_text, "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_branch"),
            },
            "solver_hits": {
                "full_solver_ell0_zero": hit(full_solver_text, "return 0.0"),
                "full_solver_charge_unity": hit(full_solver_text, "return 1.0"),
                "two_component_seed": hit(
                    two_component_solver_text,
                    "y0 = [float(amp0), 0.0, float(amp_l) * r0, float(amp_l)]",
                ),
                "two_component_amp_l_scan": hit(two_component_solver_text, "for amp_l in AMPL_GRID:"),
            },
            "next_steps_note_hits": {
                "series_first": hit(
                    next_steps_note_text,
                    "Step A. `ℓ=0` の exact ansatz と near-origin series を固定する",
                ),
                "nonzero_seed_not_first": hit(
                    next_steps_note_text,
                    "次にやるべきは nonzero seed を入れることそのものではなく",
                ),
                "success_gate": hit(next_steps_note_text, "near-origin series が nonzero `f_L` を許す"),
                "no_go_gate": hit(
                    next_steps_note_text,
                    "near-origin regularity が `f_L ≡ 0` を強制する",
                ),
            },
            "closeout_split_gate_summary": closeout_split_gate_summary,
            "route_local_gate_summary": route_local_gate_summary,
            "closure_gate_summary": closure_gate_summary,
            "recent_history_branch_line": hit(
                work_history_recent_text,
                "current-canon closeout / exploratory split contract branch",
            ),
        },
    )

    audit = payload(
        "8.7.56.1332",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory generalized-vector-solver branch audit",
        inputs,
        [
            row(
                "exploratory_generalized_vector_solver_branch_ready",
                "pass" if exploratory_generalized_vector_solver_branch_ready else "reject",
                "exploratory generalized-vector-solver branch ready",
                1 if exploratory_generalized_vector_solver_branch_ready else 0,
                "The exploratory solver branch is ready only because current-canon closeout is already frozen and the remaining work is explicitly solver-side.",
            ),
            row(
                "ell0_series_theorem_first_gate_required",
                "pass" if ell0_series_theorem_first_gate_required else "reject",
                "ell=0 series theorem first gate required",
                1 if ell0_series_theorem_first_gate_required else 0,
                "The first honest solver-side gate is the ell=0 near-origin series theorem, not a blind nonzero-seed scan.",
            ),
            row(
                "nonzero_seed_mainline_admissible_without_series",
                "reject",
                "nonzero seed mainline admissible without series",
                0,
                "A blind nonzero-seed push would overreach because the ell=0 regularity theorem is not yet fixed.",
            ),
            row(
                "solver_boundary_value_reformulation_admissible",
                "pass" if solver_boundary_value_reformulation_admissible else "reject",
                "solver boundary-value reformulation admissible",
                1 if solver_boundary_value_reformulation_admissible else 0,
                "Once the series gate is accepted, a boundary-value / operator reformulation becomes an admissible exploratory computation branch.",
            ),
            row(
                "exploratory_generalized_vector_solver_exact_computation_ready_now",
                "reject",
                "exploratory generalized-vector-solver exact computation ready now",
                0,
                "Exact vector computation remains unopened until the series theorem gate is resolved.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "This branch only opens a solver-side exploratory lane and does not require physical reject.",
            ),
        ],
        {
            "exploratory_generalized_vector_solver_branch_ready": exploratory_generalized_vector_solver_branch_ready,
            "exploratory_generalized_vector_solver_branch_honest": exploratory_generalized_vector_solver_branch_honest,
            "current_canon_already_proves_nonzero_ell0_temporal_longitudinal_closure": False,
            "ell0_series_theorem_first_gate_required": ell0_series_theorem_first_gate_required,
            "nonzero_seed_mainline_admissible_without_series": False,
            "solver_boundary_value_reformulation_admissible": solver_boundary_value_reformulation_admissible,
            "exploratory_generalized_vector_solver_exact_computation_ready_now": False,
            "solver_success_gate_ready": solver_success_gate_ready,
            "solver_no_go_gate_ready": solver_no_go_gate_ready,
            "effective_source_ansatz_branch_admissible": effective_source_ansatz_branch_candidate,
            "observable_dictionary_branch_admissible": observable_dictionary_branch_candidate,
            "physical_reject_required": False,
            "result_class": (
                "exploratory_generalized_vector_solver_branch_honest"
                if exploratory_generalized_vector_solver_branch_honest
                else "exploratory_generalized_vector_solver_branch_not_yet_honest"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_"
                "audit_completed"
            ),
            "advance_to_8_7_56_1333": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_branch_declaration_gate"
            ],
        },
        {
            "closeout_split_gate_summary": closeout_split_gate_summary,
            "closeout_split_eval_summary": closeout_split_eval_summary,
            "route_local_gate_summary": route_local_gate_summary,
            "closure_gate_summary": closure_gate_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1333",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory generalized-vector-solver branch declaration gate",
        inputs,
        [
            row(
                "exploratory_generalized_vector_solver_branch_ready",
                "pass" if exploratory_generalized_vector_solver_branch_ready else "reject",
                "exploratory generalized-vector-solver branch ready",
                1 if exploratory_generalized_vector_solver_branch_ready else 0,
                "The solver-side exploratory branch is formally admitted under the already-frozen current-canon closeout.",
            ),
            row(
                "solver_success_gate_ready",
                "pass" if solver_success_gate_ready else "reject",
                "solver success gate ready",
                1 if solver_success_gate_ready else 0,
                "Success means the ell=0 near-origin series theorem permits a nonzero or free longitudinal amplitude that can be carried to the operator stage.",
            ),
            row(
                "solver_no_go_gate_ready",
                "pass" if solver_no_go_gate_ready else "reject",
                "solver no-go gate ready",
                1 if solver_no_go_gate_ready else 0,
                "No-go means the ell=0 regularity analysis forces trivial longitudinal closure or blocks the seed from becoming a theorem-backed mode.",
            ),
            row(
                "exploratory_generalized_vector_solver_exact_computation_ready_now",
                "reject",
                "exploratory generalized-vector-solver exact computation ready now",
                0,
                "Blind vector form-factor computation remains downstream of the series theorem and operator-source stages.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "Even a solver-side no-go would be route-local and would not force physical reject of the wider program.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": closeout_split_gate_summary["trial2_numeric_alpha_problem_classification"],
            "exploratory_generalized_vector_solver_branch_ready": exploratory_generalized_vector_solver_branch_ready,
            "exploratory_generalized_vector_solver_branch_honest": exploratory_generalized_vector_solver_branch_honest,
            "current_canon_already_proves_nonzero_ell0_temporal_longitudinal_closure": False,
            "ell0_series_theorem_first_gate_required": ell0_series_theorem_first_gate_required,
            "nonzero_seed_mainline_admissible_without_series": False,
            "solver_boundary_value_reformulation_admissible": solver_boundary_value_reformulation_admissible,
            "solver_success_gate_ready": solver_success_gate_ready,
            "solver_no_go_gate_ready": solver_no_go_gate_ready,
            "exploratory_generalized_vector_solver_exact_computation_ready_now": False,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_solver_first_gate_route": ELL0_SERIES_ROUTE_NAME,
            "selected_solver_success_handoff_class": SUCCESS_HANDOFF_CLASS,
            "selected_solver_no_go_handoff_class": NO_GO_HANDOFF_CLASS,
            "selected_secondary_exploratory_route": SECONDARY_ROUTE_NAME,
            "selected_reserve_exploratory_route": RESERVE_ROUTE_NAME,
            "selected_next_generation_route": ELL0_SERIES_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_"
                "declared"
            ),
            "advance_to_8_7_56_1334": True,
            "next_required_artifacts": [ELL0_SERIES_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "closeout_split_eval_summary": closeout_split_eval_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1334",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory generalized-vector-solver branch numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float_value(closeout_split_eval_summary, "beta_1"),
                "The exploratory generalized-vector-solver branch keeps the retained beta_1 baseline fixed.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float_value(closeout_split_eval_summary, "q_theory_over_m0"),
                "The exploratory generalized-vector-solver branch keeps the retained matching-scale baseline fixed.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float_value(closeout_split_eval_summary, "F_exact_at_q_theory"),
                "The exploratory generalized-vector-solver branch keeps the retained exact-profile overlap baseline fixed.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha_exact at q_theory fixed",
                float_value(closeout_split_eval_summary, "alpha_exact_at_q_theory"),
                "The exploratory generalized-vector-solver branch keeps the retained alpha baseline fixed.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float_value(closeout_split_eval_summary, "exact_ground_state_polarization_weight"),
                "The current full solver still keeps the exact ell=0 ground state at zero polarization weight before the exploratory series theorem gate.",
            ),
            row(
                "ell0_zero_seed_max_abs_fL_fixed",
                "pass",
                "ell=0 zero-seed max abs fL fixed",
                float_value(closeout_split_eval_summary, "ell0_zero_seed_max_abs_fL"),
                "The current retained zero-seed longitudinal amplitude remains zero before the exploratory solver redesign.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "reject",
                "numeric state changed by current branch",
                0,
                "This branch only freezes the first exploratory solver gate and does not create a new vector numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1,
                "The route now advances from generic exploratory split into the specific ell=0 series theorem first-gate branch.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1": float_value(closeout_split_eval_summary, "beta_1"),
            "exact_ground_state_polarization_weight": float_value(
                closeout_split_eval_summary,
                "exact_ground_state_polarization_weight",
            ),
            "exact_ground_state_coupled_charge_factor": float_value(
                closeout_split_eval_summary,
                "exact_ground_state_coupled_charge_factor",
            ),
            "ell0_zero_seed_max_abs_fL": float_value(
                closeout_split_eval_summary,
                "ell0_zero_seed_max_abs_fL",
            ),
            "q_theory_over_m0": float_value(closeout_split_eval_summary, "q_theory_over_m0"),
            "F_exact_at_q_theory": float_value(closeout_split_eval_summary, "F_exact_at_q_theory"),
            "alpha_exact_at_q_theory": float_value(
                closeout_split_eval_summary,
                "alpha_exact_at_q_theory",
            ),
            "exploratory_generalized_vector_solver_branch_completed": exploratory_generalized_vector_solver_branch_ready,
            "ell0_series_theorem_first_gate_required": ell0_series_theorem_first_gate_required,
            "exploratory_generalized_vector_solver_exact_computation_ready_now": False,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": ELL0_SERIES_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_"
                "branch_completed"
            ),
            "advance_to_next_route": True,
            "next_required_artifacts": [ELL0_SERIES_ROUTE_NAME],
        },
        {
            "prior_problem_classification": closeout_split_gate_summary["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": BRANCH_CLASS,
            "closeout_split_eval_summary": closeout_split_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_branch_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_branch_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_branch_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_branch_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1331-.1334 artifacts generated")


if __name__ == "__main__":
    main()
