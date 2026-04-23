#!/usr/bin/env python3
"""Generate 8.7.56.1275-.1278 vector Q-ball form-factor review artifacts.

Purpose:
    Re-check the newly proposed "vector Q-ball form factor" viewpoint before
    freezing the route-local no-go carry-over contract. The key question is
    whether the claimed missing piece is already embedded in the current
    exact/pilot machinery, or whether the note instead exposes a new
    computation-side gap: the lack of a ground-state two-component closure.

Inputs:
    - Current operational docs and the current Trial-2 problem/status notes
    - The new note
      `C:/Users/ogawa/Downloads/pmodel_v2_trial2_vector_qball_form_factor.md`
    - The already frozen route-local no-go theorem review metrics
    - The retained scalar ground-state metrics
    - The retained exact vector full-coupled pilot metrics
    - The existing Trial-3 two-component pivot/spectrum solver scripts

Outputs:
    - Four machine-readable metrics payloads under `output/public/quantum/`

Assumptions:
    - The current exact vector ground state remains the retained
      `(n,k,ell,s)=(1,0,0,0)` reference state from the exact coupled ladder.
    - The current two-component pilot is allowed only as a smoke test; it does
      not itself prove an electron ground-state longitudinal component unless
      the ell=0 coupling is actually present.
"""

from __future__ import annotations

import csv
import importlib.util
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

NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_vector_qball_form_factor.md")
ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "declaration_gate_metrics.json"
)
ROUTE_LOCAL_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "numeric_evaluation_metrics.json"
)
SCALAR_GROUND_STATE = PUBLIC_OUT / "mass_origin_qball_discrete_mass_spectrum_metrics.json"
EXACT_FULL_COUPLED = PUBLIC_OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"
SCALAR_OVERLAP_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_numeric_evaluation_metrics.json"

PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"
SPECTRUM_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.1279"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort if one required input is missing.

def require(path: Path) -> None:
    """Abort if one required input is missing."""
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


# Function: build one standard metrics payload.

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


# Function: write one JSON metrics payload and one CSV row table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one JSON metrics payload and one CSV row table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: load one local Python module.

def load_module(path: Path, module_name: str):
    """Load one local Python module."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: extract the retained scalar ground-state row.

def extract_scalar_ground_state(data: dict) -> dict:
    """Extract the retained scalar ground-state row."""
    for row_data in data["evidence"]["discrete_mass_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "mode_index": int(row_data["mode_index"]),
                "beta_n": float(row_data["beta_n"]),
                "charge_proxy": float(row_data["charge_proxy"]),
                "energy_proxy": float(row_data["energy_proxy"]),
                "central_amplitude": float(row_data["central_amplitude"]),
            }

    raise SystemExit("[fail] missing scalar ground-state row")


# Function: extract the retained exact vector ground-state row.

def extract_exact_ground_state(data: dict) -> dict:
    """Extract the retained exact vector ground-state row."""
    for row_data in data["evidence"]["exact_ladder_sample_rows"]:
        if (
            int(row_data["n"]) == 1
            and int(row_data["k"]) == 0
            and int(row_data["ell"]) == 0
            and int(row_data["s"]) == 0
        ):
            return {
                "n": int(row_data["n"]),
                "k": int(row_data["k"]),
                "ell": int(row_data["ell"]),
                "s": int(row_data["s"]),
                "beta_n": float(row_data["beta_n"]),
                "polarization_weight": float(row_data["polarization_weight"]),
                "coupled_charge_factor": float(row_data["coupled_charge_factor"]),
                "coupled_mass_factor": float(row_data["coupled_mass_factor"]),
            }

    raise SystemExit("[fail] missing exact ground-state row")


# Function: run one ell=0 two-component smoke test with the retained beta_1 state.

def ell0_smoke(spec_module, pivot_module, numerical_module, beta_1: float, amp0: float, amp_l: float) -> dict:
    """Run one ell=0 two-component smoke test with the retained beta_1 state."""
    result = spec_module.solve_two_component_profile(
        pivot_module,
        numerical_module,
        float(beta_1),
        0,
        float(amp0),
        float(amp_l),
    )
    return {
        "beta": float(result["beta"]),
        "amp0": float(result["amp0"]),
        "amp_l": float(result["amp_l"]),
        "tail_to_input_ratio": float(result["tail_to_input_ratio"]),
        "max_abs_fL": float(result["max_abs_fL"]),
        "final_fL": float(result["final_fL"]),
    }


# Function: execute the 8.7.56.1275-.1278 branch.

def main() -> None:
    """Execute the 8.7.56.1275-.1278 branch."""
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
        NOTE,
        ROUTE_LOCAL_GATE,
        ROUTE_LOCAL_EVAL,
        SCALAR_GROUND_STATE,
        EXACT_FULL_COUPLED,
        SCALAR_OVERLAP_EVAL,
        PIVOT_BRANCH,
        SPECTRUM_BRANCH,
        NUMERICAL_BRANCH,
        FULL_BRANCH,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    note_text = read_text(NOTE)

    route_local_gate = read_json(ROUTE_LOCAL_GATE)
    route_local_eval = read_json(ROUTE_LOCAL_EVAL)
    scalar_ground_state_metrics = read_json(SCALAR_GROUND_STATE)
    exact_full_coupled_metrics = read_json(EXACT_FULL_COUPLED)
    scalar_overlap_eval = read_json(SCALAR_OVERLAP_EVAL)

    pivot_module = load_module(PIVOT_BRANCH, "wavep_t3_pivot")
    spectrum_module = load_module(SPECTRUM_BRANCH, "wavep_t3_spectrum")
    numerical_module = load_module(NUMERICAL_BRANCH, "wavep_vector_qball_numerical")
    full_module = load_module(FULL_BRANCH, "wavep_vector_qball_full")

    scalar_ground_state = extract_scalar_ground_state(scalar_ground_state_metrics)
    exact_ground_state = extract_exact_ground_state(exact_full_coupled_metrics)

    beta_1 = float(scalar_ground_state["beta_n"])
    amp0 = float(scalar_ground_state["central_amplitude"])
    ell0_zero_seed = ell0_smoke(spectrum_module, pivot_module, numerical_module, beta_1, amp0, 0.0)
    ell0_matched_seed = ell0_smoke(spectrum_module, pivot_module, numerical_module, beta_1, amp0, amp0)

    ell0_off_diagonal_coupling_available = False
    exact_ground_state_reduces_to_scalar_reference = (
        abs(float(exact_ground_state["polarization_weight"])) == 0.0
        and abs(float(exact_ground_state["coupled_charge_factor"]) - 1.0) == 0.0
        and abs(float(exact_ground_state["coupled_mass_factor"]) - 1.0) == 0.0
    )
    zero_seed_keeps_longitudinal_component_zero = (
        abs(float(ell0_zero_seed["max_abs_fL"])) == 0.0
        and abs(float(ell0_zero_seed["final_fL"])) == 0.0
    )
    manual_seed_required_for_nonzero_longitudinal_component = (
        abs(float(ell0_matched_seed["max_abs_fL"])) > 0.0
        and zero_seed_keeps_longitudinal_component_zero
    )

    vector_form_factor_candidate_admissible = True
    vector_form_factor_exact_computation_ready_under_current_pack = False
    vector_form_factor_note_already_embedded_in_current_solver = False
    route_local_no_go_theorem_retained = (
        route_local_gate["summary"]["trial2_numeric_alpha_problem_classification"]
        == "qball_projection_overlap_route_local_no_go_theorem_under_current_canon"
    )
    carry_over_contract_branch_retired = True
    literal_q_equals_m0_reopened_under_current_pack = False
    ground_state_two_component_closure_required = True

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
            "vector_form_factor_note": display_path(NOTE),
        },
        "prior_metrics": {
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
            "route_local_eval": display_path(ROUTE_LOCAL_EVAL),
            "scalar_ground_state": display_path(SCALAR_GROUND_STATE),
            "exact_full_coupled": display_path(EXACT_FULL_COUPLED),
            "scalar_overlap_eval": display_path(SCALAR_OVERLAP_EVAL),
        },
        "solver_modules": {
            "pivot_branch": display_path(PIVOT_BRANCH),
            "spectrum_branch": display_path(SPECTRUM_BRANCH),
            "numerical_branch": display_path(NUMERICAL_BRANCH),
            "full_branch": display_path(FULL_BRANCH),
        },
        "constants": {
            "beta_1": beta_1,
            "amp0": amp0,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1275",
        "Trial-2 numeric alpha vector Q-ball form-factor review source inventory",
        inputs,
        [
            row("vector_form_factor_note_available", "pass", "vector form-factor note available", 1.0, "The new note explicitly argues that the observable should be a vector signed charge density rather than the old scalar |f|^2 density."),
            row("retained_scalar_ground_state_available", "pass", "retained scalar ground state available", 1.0, "The frozen beta_1 and the scalar central amplitude are still available for any ground-state closure test."),
            row("retained_exact_vector_ground_state_available", "pass", "retained exact vector ground state available", 1.0, "The exact coupled ladder still exposes the current vector ground-state reference row."),
            row("existing_two_component_solver_available", "pass", "existing two-component solver available", 1.0, "The Trial-3 two-component pivot/spectrum solver exists and can be smoke-tested immediately."),
            row("route_local_no_go_state_available", "pass" if route_local_no_go_theorem_retained else "reject", "route-local no-go state available", 1 if route_local_no_go_theorem_retained else 0, "The new review starts from the already frozen T2 route-local no-go theorem state."),
        ],
        {
            "inventory_ready": True,
            "selected_next_substep": "8.7.56.1276",
            "prior_problem_classification": route_local_gate["summary"]["trial2_numeric_alpha_problem_classification"],
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_inventory_fixed",
            "advance_to_8_7_56_1276": True,
            "next_required_artifacts": ["trial2_numeric_alpha_vector_qball_form_factor_review_audit"],
        },
        {
            "note_hits": {
                "vector_noether_charge_density": hit(note_text, "vector Q-ball の Noether charge density"),
                "signed_density_line": hit(note_text, r"j^0_{\rm vector} = 2\omega"),
                "q_equals_m0_claim": hit(note_text, "q = m_0"),
                "computation_plan": hit(note_text, "## computation plan"),
            },
            "solver_hits": {
                "ell0_kproxy_line": hit(read_text(SPECTRUM_BRANCH), "k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr"),
                "ell0_polarization_zero_line": hit(read_text(FULL_BRANCH), "if ell == 0:"),
                "pivot_boundary_conditions": hit(read_text(PIVOT_BRANCH), "boundary_conditions"),
            },
            "status_hits": {
                "status_1275": hit(status_text, "8.7.56.1275"),
                "roadmap_1275": hit(roadmap_text, "`8.7.56.1275-.1278`"),
                "recent_1271": hit(work_history_recent_text, "8.7.56.1271-.1274"),
                "problem_route_local": hit(current_problem_text, "route-local no-go"),
                "status_route_local": hit(current_status_text, "route-local no-go"),
            },
            "part_hits": {
                "part1_two_component": hit(part1_text, "two-component closeout"),
                "part1_reference_state": hit(part1_text, r"m_0 = \frac{m_e}{\mathcal{E}(\beta_1)}"),
                "part3a_vector_qball": hit(part3a_text, "vector-Q-ball"),
                "part5_trial2_table": hit(part5_text, "| Trial-2 |"),
            },
        },
    )

    audit = payload(
        "8.7.56.1276",
        "Trial-2 numeric alpha vector Q-ball form-factor review audit",
        inputs,
        [
            row("vector_form_factor_candidate_admissible", "pass" if vector_form_factor_candidate_admissible else "reject", "vector form-factor candidate admissible", 1 if vector_form_factor_candidate_admissible else 0, "The note identifies a concrete observable-side computation candidate rather than another wording-only retry."),
            row("current_exact_ground_state_reduces_to_scalar_reference", "pass" if exact_ground_state_reduces_to_scalar_reference else "reject", "current exact ground state reduces to scalar reference", 1 if exact_ground_state_reduces_to_scalar_reference else 0, "The retained exact vector ground state still carries zero polarization weight and unit charge/mass factors."),
            row("ell0_off_diagonal_coupling_available_in_existing_solver", "pass" if ell0_off_diagonal_coupling_available else "reject", "ell=0 off-diagonal coupling available in existing solver", 1 if ell0_off_diagonal_coupling_available else 0, "The existing two-component pilot uses C_ell proportional to sqrt(ell(ell+1))/r, so ell=0 currently has no off-diagonal temporal/longitudinal mixing."),
            row("ell0_zero_seed_keeps_longitudinal_component_zero", "pass" if zero_seed_keeps_longitudinal_component_zero else "reject", "ell=0 zero seed keeps longitudinal component zero", 1 if zero_seed_keeps_longitudinal_component_zero else 0, "With amp_l = 0, the current ell=0 pilot keeps f_L identically zero."),
            row("ell0_manual_seed_required_for_nonzero_longitudinal_component", "pass" if manual_seed_required_for_nonzero_longitudinal_component else "reject", "ell=0 manual seed required for nonzero longitudinal component", 1 if manual_seed_required_for_nonzero_longitudinal_component else 0, "Any nonzero ell=0 longitudinal component currently appears only after an explicit manual seed, not by induction from f_0."),
            row("vector_form_factor_note_already_embedded_in_current_solver", "pass" if vector_form_factor_note_already_embedded_in_current_solver else "reject", "vector form-factor note already embedded in current solver", 1 if vector_form_factor_note_already_embedded_in_current_solver else 0, "The note would only be 'already embedded' if the current exact/pilot machinery already produced a nontrivial ell=0 longitudinal component."),
            row("vector_form_factor_exact_computation_ready_under_current_pack", "pass" if vector_form_factor_exact_computation_ready_under_current_pack else "reject", "vector form-factor exact computation ready under current pack", 1 if vector_form_factor_exact_computation_ready_under_current_pack else 0, "An honest exact computation would need a canonical ground-state two-component closure, which the current pack does not yet provide."),
            row("route_local_no_go_theorem_retained", "pass" if route_local_no_go_theorem_retained else "reject", "route-local no-go theorem retained", 1 if route_local_no_go_theorem_retained else 0, "The new note does not itself overturn the already frozen T2 route-local no-go theorem."),
        ],
        {
            "vector_form_factor_candidate_admissible": vector_form_factor_candidate_admissible,
            "current_exact_ground_state_reduces_to_scalar_reference": exact_ground_state_reduces_to_scalar_reference,
            "ell0_off_diagonal_coupling_available_in_existing_solver": ell0_off_diagonal_coupling_available,
            "ell0_zero_seed_keeps_longitudinal_component_zero": zero_seed_keeps_longitudinal_component_zero,
            "ell0_manual_seed_required_for_nonzero_longitudinal_component": manual_seed_required_for_nonzero_longitudinal_component,
            "vector_form_factor_note_already_embedded_in_current_solver": vector_form_factor_note_already_embedded_in_current_solver,
            "vector_form_factor_exact_computation_ready_under_current_pack": vector_form_factor_exact_computation_ready_under_current_pack,
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "result_class": "vector_form_factor_candidate_reveals_ground_state_two_component_closure_gap",
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_audit_completed",
            "advance_to_8_7_56_1277": True,
            "next_required_artifacts": ["trial2_numeric_alpha_vector_qball_form_factor_review_declaration_gate"],
        },
        {
            "scalar_ground_state": scalar_ground_state,
            "exact_ground_state": exact_ground_state,
            "ell0_zero_seed_smoke": ell0_zero_seed,
            "ell0_matched_seed_smoke": ell0_matched_seed,
            "route_local_gate_summary": route_local_gate["summary"],
            "route_local_eval_summary": route_local_eval["summary"],
            "scalar_overlap_eval_summary": scalar_overlap_eval["summary"],
        },
    )

    declaration_gate = payload(
        "8.7.56.1277",
        "Trial-2 numeric alpha vector Q-ball form-factor review declaration gate",
        inputs,
        [
            row("vector_form_factor_review_completed", "pass", "vector form-factor review completed", 1.0, "The new observable-side computation note has been audited against the retained exact and pilot machinery."),
            row("carry_over_contract_branch_retired", "pass" if carry_over_contract_branch_retired else "reject", "carry-over contract branch retired", 1 if carry_over_contract_branch_retired else 0, "The old carry-over contract plan is paused because the note exposes an unresolved observable-side closure gap worth checking first."),
            row("ground_state_two_component_closure_required", "pass" if ground_state_two_component_closure_required else "reject", "ground-state two-component closure required", 1 if ground_state_two_component_closure_required else 0, "The immediate next question is whether the electron-like ground state truly carries a canonical two-component temporal/longitudinal structure."),
            row("route_local_no_go_theorem_retained", "pass" if route_local_no_go_theorem_retained else "reject", "route-local no-go theorem retained", 1 if route_local_no_go_theorem_retained else 0, "The T2 route-local no-go remains frozen under current canon."),
            row("literal_q_equals_m0_reopened_under_current_pack", "pass" if literal_q_equals_m0_reopened_under_current_pack else "reject", "literal q = m0 reopened under current pack", 1 if literal_q_equals_m0_reopened_under_current_pack else 0, "The note does not yet reopen literal q = m0 under the currently frozen exact/pilot machinery."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "This remains a computation-side closure gap, not a physical reject."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_required",
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "vector_form_factor_candidate_admissible": vector_form_factor_candidate_admissible,
            "current_exact_ground_state_reduces_to_scalar_reference": exact_ground_state_reduces_to_scalar_reference,
            "ground_state_two_component_closure_required": ground_state_two_component_closure_required,
            "vector_form_factor_exact_computation_ready_under_current_pack": vector_form_factor_exact_computation_ready_under_current_pack,
            "primary_residual_lane": "vector_qball_form_factor_ground_state_two_component_closure",
            "secondary_residual_lane": "qball_projection_overlap_future_source_theorem_reopen",
            "reserve_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_pivot_declared",
            "advance_to_8_7_56_1278": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "note_hits": inventory["evidence"]["note_hits"],
        },
    )

    evaluation = payload(
        "8.7.56.1278",
        "Trial-2 numeric alpha vector Q-ball form-factor review numeric evaluation",
        inputs,
        [
            row("beta_1_fixed", "pass", "beta_1 fixed", beta_1, "The retained electron-like ground-state beta_1 remains frozen."),
            row("exact_ground_state_polarization_weight_fixed", "pass", "exact ground-state polarization weight fixed", float(exact_ground_state["polarization_weight"]), "The retained exact vector ground state currently carries zero polarization weight."),
            row("exact_ground_state_coupled_charge_factor_fixed", "pass", "exact ground-state coupled charge factor fixed", float(exact_ground_state["coupled_charge_factor"]), "The retained exact vector ground state currently keeps a unit coupled charge factor."),
            row("ell0_zero_seed_max_abs_fL_fixed", "pass", "ell=0 zero-seed max |f_L| fixed", float(ell0_zero_seed["max_abs_fL"]), "The current two-component pilot keeps the longitudinal component identically zero when the ell=0 seed is zero."),
            row("ell0_matched_seed_max_abs_fL_fixed", "pass", "ell=0 matched-seed max |f_L| fixed", float(ell0_matched_seed["max_abs_fL"]), "A nonzero ell=0 longitudinal component appears only after a manual matched seed."),
            row("scalar_literal_F_m0_fixed", "pass", "scalar literal F(m0) fixed", float(scalar_overlap_eval["summary"]["F_m0"]), "The retained scalar overlap still fails at literal q = m0 and therefore remains the current exact baseline."),
            row("numeric_state_changed_by_current_branch", "reject", "numeric state changed by current branch", 0.0, "This branch only reclassifies the observable-side closure gap; it does not yet produce a new exact alpha candidate."),
            row("route_state_changed_by_current_branch", "pass", "route state changed by current branch", 1.0, "The route shifts from a pure carry-over contract plan to a ground-state two-component closure review."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_required",
            "beta_1": beta_1,
            "exact_ground_state_polarization_weight": float(exact_ground_state["polarization_weight"]),
            "exact_ground_state_coupled_charge_factor": float(exact_ground_state["coupled_charge_factor"]),
            "ell0_zero_seed_max_abs_fL": float(ell0_zero_seed["max_abs_fL"]),
            "ell0_matched_seed_max_abs_fL": float(ell0_matched_seed["max_abs_fL"]),
            "scalar_literal_F_m0": float(scalar_overlap_eval["summary"]["F_m0"]),
            "vector_form_factor_exact_computation_ready_under_current_pack": vector_form_factor_exact_computation_ready_under_current_pack,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_review_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": route_local_gate["summary"]["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_required",
        },
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_review_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_review_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_review_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_review_numeric_evaluation", evaluation)

    print("[done] 8.7.56.1275-.1278 artifacts generated")


if __name__ == "__main__":
    main()
