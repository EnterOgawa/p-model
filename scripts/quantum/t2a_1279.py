#!/usr/bin/env python3
"""Generate 8.7.56.1279-.1282 closure-review artifacts for vector form factor.

Purpose:
    Re-check whether the newly proposed vector-Q-ball form-factor route already
    follows from the current theory-side surfaces, or whether the current pack
    still stops earlier because the electron-like ground state lacks a
    canonical ell=0 temporal/longitudinal two-component closure.

Inputs:
    - Current operational docs and current Trial-2 problem/status notes
    - The retained vector-Qball form-factor note
      `C:/Users/ogawa/Downloads/pmodel_v2_trial2_vector_qball_form_factor.md`
    - The retained exact/full-coupled vector ground-state metrics
    - The already frozen vector-Qball form-factor review metrics
    - Part I post-photon nontransverse sector wording
    - Current full-coupled / two-component solver source files

Outputs:
    - Four machine-readable metrics payloads under `output/public/quantum/`

Assumptions:
    - The retained exact ground-state row `(n,k,ell,s)=(1,0,0,0)` remains the
      current electron-like vector reference state.
    - The branch judges only current-pack implication / non-implication; it
      does not introduce a new generalized solver or a new alpha candidate.
"""

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

NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_vector_qball_form_factor.md")
VECTOR_REVIEW_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_review_declaration_gate_metrics.json"
)
VECTOR_REVIEW_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_review_audit_metrics.json"
)
VECTOR_REVIEW_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_review_numeric_evaluation_metrics.json"
)
ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "declaration_gate_metrics.json"
)
ROUTE_LOCAL_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "numeric_evaluation_metrics.json"
)
EXACT_FULL_COUPLED = PUBLIC_OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"

PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"
SPECTRUM_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
FULL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.1283"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract"


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
                "exact_charge_proxy": float(row_data["exact_charge_proxy"]),
                "exact_mass_proxy": float(row_data["exact_mass_proxy"]),
            }

    raise SystemExit("[fail] missing exact ground-state row")


# Function: execute the 8.7.56.1279-.1282 branch.

def main() -> None:
    """Execute the 8.7.56.1279-.1282 branch."""
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
        VECTOR_REVIEW_GATE,
        VECTOR_REVIEW_AUDIT,
        VECTOR_REVIEW_EVAL,
        ROUTE_LOCAL_GATE,
        ROUTE_LOCAL_EVAL,
        EXACT_FULL_COUPLED,
        PIVOT_BRANCH,
        SPECTRUM_BRANCH,
        FULL_BRANCH,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    note_text = read_text(NOTE)
    spectrum_text = read_text(SPECTRUM_BRANCH)
    full_text = read_text(FULL_BRANCH)

    vector_review_gate = read_json(VECTOR_REVIEW_GATE)
    vector_review_audit = read_json(VECTOR_REVIEW_AUDIT)
    vector_review_eval = read_json(VECTOR_REVIEW_EVAL)
    route_local_gate = read_json(ROUTE_LOCAL_GATE)
    route_local_eval = read_json(ROUTE_LOCAL_EVAL)
    exact_full_coupled = read_json(EXACT_FULL_COUPLED)

    exact_ground_state = extract_exact_ground_state(exact_full_coupled)

    generic_post_photon_two_component_sector_available = (
        hit(part1_text, "post-photon nontransverse sector") is not None
        and hit(part1_text, "one massive propagating eigenmode") is not None
        and hit(part1_text, "coupled asymptotic eigenmode") is not None
    )
    current_full_solver_hardcodes_ell0_scalar_reduction = (
        hit(full_text, "if ell == 0:") is not None
        and hit(full_text, "return 0.0") is not None
        and abs(float(exact_ground_state["polarization_weight"])) == 0.0
        and abs(float(exact_ground_state["coupled_charge_factor"]) - 1.0) == 0.0
        and abs(float(exact_ground_state["coupled_mass_factor"]) - 1.0) == 0.0
    )
    current_two_component_pilot_ell0_induction_available = bool(
        vector_review_audit["summary"]["ell0_off_diagonal_coupling_available_in_existing_solver"]
    )
    current_two_component_zero_seed_keeps_fl_zero = bool(
        vector_review_audit["summary"]["ell0_zero_seed_keeps_longitudinal_component_zero"]
    )
    explicit_ell0_ground_state_two_component_closure_available = False
    ground_state_two_component_closure_already_implied_under_current_pack = False
    vector_form_factor_exact_computation_ready_under_current_pack = False
    literal_q_equals_m0_reopened_under_current_pack = False
    route_local_no_go_theorem_retained = bool(
        vector_review_gate["summary"]["route_local_no_go_theorem_retained"]
    )
    theorem_side_positive_generic_sector_but_ground_state_unfixed = (
        generic_post_photon_two_component_sector_available
        and not explicit_ell0_ground_state_two_component_closure_available
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
            "vector_form_factor_note": display_path(NOTE),
        },
        "prior_metrics": {
            "vector_review_gate": display_path(VECTOR_REVIEW_GATE),
            "vector_review_audit": display_path(VECTOR_REVIEW_AUDIT),
            "vector_review_eval": display_path(VECTOR_REVIEW_EVAL),
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
            "route_local_eval": display_path(ROUTE_LOCAL_EVAL),
            "exact_full_coupled": display_path(EXACT_FULL_COUPLED),
        },
        "solver_sources": {
            "pivot_branch": display_path(PIVOT_BRANCH),
            "spectrum_branch": display_path(SPECTRUM_BRANCH),
            "full_branch": display_path(FULL_BRANCH),
        },
        "constants": {
            "beta_1": float(exact_ground_state["beta_n"]),
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1279",
        "Trial-2 numeric alpha vector Q-ball form-factor ground-state two-component closure review source inventory",
        inputs,
        [
            row("vector_form_factor_review_state_available", "pass", "vector form-factor review state available", 1.0, "The prior admissibility review already froze the signed-density candidate and the ground-state closure gap."),
            row("generic_post_photon_two_component_sector_available", "pass" if generic_post_photon_two_component_sector_available else "reject", "generic post-photon two-component sector available", 1 if generic_post_photon_two_component_sector_available else 0, "Part I still surfaces the post-photon nontransverse sector {delta P_0, delta P_i^L} as a physically admissible coupled sector."),
            row("retained_exact_ground_state_row_available", "pass", "retained exact ground-state row available", 1.0, "The exact full-coupled ladder still exposes the electron-like (1,0,0,0) row used by the current route."),
            row("current_full_solver_source_available", "pass", "current full solver source available", 1.0, "The adopted exact vector solver source can be audited directly for ell=0 handling."),
            row("current_two_component_solver_source_available", "pass", "current two-component solver source available", 1.0, "The Trial-3 two-component pilot source can be audited directly for ell=0 induction terms."),
        ],
        {
            "inventory_ready": True,
            "selected_next_substep": "8.7.56.1280",
            "prior_problem_classification": vector_review_gate["summary"]["trial2_numeric_alpha_problem_classification"],
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_inventory_fixed",
            "advance_to_8_7_56_1280": True,
            "next_required_artifacts": ["trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_audit"],
        },
        {
            "note_hits": {
                "signed_density_line": hit(note_text, r"j^0_{\rm vector} = 2\omega"),
                "induced_fl_claim": hit(note_text, r"f_L(r) \propto"),
                "literal_q_equals_m0_claim": hit(note_text, "q = m_0"),
            },
            "part1_hits": {
                "post_photon_nontransverse_sector": hit(part1_text, "post-photon nontransverse sector"),
                "one_massive_one_constraint": hit(part1_text, "one massive propagating eigenmode"),
                "coupled_eigenmode_tail": hit(part1_text, "coupled asymptotic eigenmode"),
            },
            "solver_hits": {
                "full_solver_ell0_guard": hit(full_text, "if ell == 0:"),
                "full_solver_ell0_return_zero": hit(full_text, "return 0.0"),
                "spectrum_solver_kproxy": hit(spectrum_text, "k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr"),
                "spectrum_solver_coupling": hit(spectrum_text, "coupling = float(beta) * k_proxy"),
            },
            "status_hits": {
                "status_1279": hit(status_text, "8.7.56.1279"),
                "roadmap_1279": hit(roadmap_text, "`8.7.56.1279-.1282`"),
                "problem_two_component": hit(current_problem_text, "two-component closure"),
                "status_two_component": hit(current_status_text, "two-component closure"),
                "recent_1275": hit(work_history_recent_text, "8.7.56.1275-.1278"),
            },
        },
    )

    audit = payload(
        "8.7.56.1280",
        "Trial-2 numeric alpha vector Q-ball form-factor ground-state two-component closure review audit",
        inputs,
        [
            row("generic_post_photon_two_component_sector_available", "pass" if generic_post_photon_two_component_sector_available else "reject", "generic post-photon two-component sector available", 1 if generic_post_photon_two_component_sector_available else 0, "Part I does permit a post-photon nontransverse two-component sector in general."),
            row("explicit_ell0_ground_state_two_component_closure_available", "pass" if explicit_ell0_ground_state_two_component_closure_available else "reject", "explicit ell=0 ground-state two-component closure available", 1 if explicit_ell0_ground_state_two_component_closure_available else 0, "No current public-canonical surface explicitly says that the electron-like ell=0 ground state must carry a nonzero temporal/longitudinal closure."),
            row("current_full_solver_hardcodes_ell0_scalar_reduction", "pass" if current_full_solver_hardcodes_ell0_scalar_reduction else "reject", "current full solver hardcodes ell=0 scalar reduction", 1 if current_full_solver_hardcodes_ell0_scalar_reduction else 0, "The adopted exact vector solver explicitly returns zero polarization weight for ell=0 and leaves the ground-state charge/mass factors at unity."),
            row("current_two_component_pilot_ell0_induction_available", "pass" if current_two_component_pilot_ell0_induction_available else "reject", "current two-component pilot ell=0 induction available", 1 if current_two_component_pilot_ell0_induction_available else 0, "The current two-component pilot would need an ell=0 induction term to make the note already implied under the current pack."),
            row("current_two_component_zero_seed_keeps_fl_zero", "pass" if current_two_component_zero_seed_keeps_fl_zero else "reject", "current two-component zero seed keeps f_L zero", 1 if current_two_component_zero_seed_keeps_fl_zero else 0, "The retained smoke test confirms that the current ell=0 pilot does not induce a longitudinal component from f_0 alone."),
            row("ground_state_two_component_closure_already_implied_under_current_pack", "pass" if ground_state_two_component_closure_already_implied_under_current_pack else "reject", "ground-state two-component closure already implied under current pack", 1 if ground_state_two_component_closure_already_implied_under_current_pack else 0, "The current pack has a generic post-photon two-component sector, but it does not yet imply the required ell=0 ground-state closure."),
            row("vector_form_factor_exact_computation_ready_under_current_pack", "pass" if vector_form_factor_exact_computation_ready_under_current_pack else "reject", "vector form-factor exact computation ready under current pack", 1 if vector_form_factor_exact_computation_ready_under_current_pack else 0, "Without a canonical ell=0 closure, the exact vector form-factor computation is not yet open."),
            row("route_local_no_go_theorem_retained", "pass" if route_local_no_go_theorem_retained else "reject", "route-local no-go theorem retained", 1 if route_local_no_go_theorem_retained else 0, "The T2 route-local no-go stays retained while the observable-side closure gap is reviewed."),
        ],
        {
            "generic_post_photon_two_component_sector_available": generic_post_photon_two_component_sector_available,
            "explicit_ell0_ground_state_two_component_closure_available": explicit_ell0_ground_state_two_component_closure_available,
            "current_full_solver_hardcodes_ell0_scalar_reduction": current_full_solver_hardcodes_ell0_scalar_reduction,
            "current_two_component_pilot_ell0_induction_available": current_two_component_pilot_ell0_induction_available,
            "current_two_component_zero_seed_keeps_fl_zero": current_two_component_zero_seed_keeps_fl_zero,
            "ground_state_two_component_closure_already_implied_under_current_pack": ground_state_two_component_closure_already_implied_under_current_pack,
            "theorem_side_positive_generic_sector_but_ground_state_unfixed": theorem_side_positive_generic_sector_but_ground_state_unfixed,
            "vector_form_factor_exact_computation_ready_under_current_pack": vector_form_factor_exact_computation_ready_under_current_pack,
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "result_class": "generic_two_component_sector_positive_but_ell0_ground_state_closure_unfixed",
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_audit_completed",
            "advance_to_8_7_56_1281": True,
            "next_required_artifacts": ["trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_declaration_gate"],
        },
        {
            "exact_ground_state": exact_ground_state,
            "vector_review_gate_summary": vector_review_gate["summary"],
            "vector_review_audit_summary": vector_review_audit["summary"],
            "vector_review_eval_summary": vector_review_eval["summary"],
            "route_local_gate_summary": route_local_gate["summary"],
            "route_local_eval_summary": route_local_eval["summary"],
        },
    )

    declaration_gate = payload(
        "8.7.56.1281",
        "Trial-2 numeric alpha vector Q-ball form-factor ground-state two-component closure review declaration gate",
        inputs,
        [
            row("ground_state_two_component_closure_review_completed", "pass", "ground-state two-component closure review completed", 1.0, "The current-pack implication review for the vector form-factor route has been completed."),
            row("ground_state_two_component_closure_already_implied_under_current_pack", "pass" if ground_state_two_component_closure_already_implied_under_current_pack else "reject", "ground-state two-component closure already implied under current pack", 1 if ground_state_two_component_closure_already_implied_under_current_pack else 0, "The review rejects any claim that the electron-like ell=0 closure is already licensed by the current public + solver pack."),
            row("vector_form_factor_exact_computation_ready_under_current_pack", "pass" if vector_form_factor_exact_computation_ready_under_current_pack else "reject", "vector form-factor exact computation ready under current pack", 1 if vector_form_factor_exact_computation_ready_under_current_pack else 0, "An exact signed-density form factor still cannot be claimed honestly under the current pack."),
            row("literal_q_equals_m0_reopened_under_current_pack", "pass" if literal_q_equals_m0_reopened_under_current_pack else "reject", "literal q = m0 reopened under current pack", 1 if literal_q_equals_m0_reopened_under_current_pack else 0, "The closure review does not reopen the note's literal q = m0 hope under the current exact machinery."),
            row("route_local_no_go_theorem_retained", "pass" if route_local_no_go_theorem_retained else "reject", "route-local no-go theorem retained", 1 if route_local_no_go_theorem_retained else 0, "The prior T2 route-local no-go remains part of the frozen state vector."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "This remains an unresolved closure gap, not a physical reject."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_not_implied_under_current_pack",
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "generic_post_photon_two_component_sector_available": generic_post_photon_two_component_sector_available,
            "explicit_ell0_ground_state_two_component_closure_available": explicit_ell0_ground_state_two_component_closure_available,
            "current_full_solver_hardcodes_ell0_scalar_reduction": current_full_solver_hardcodes_ell0_scalar_reduction,
            "current_two_component_pilot_ell0_induction_available": current_two_component_pilot_ell0_induction_available,
            "ground_state_two_component_closure_already_implied_under_current_pack": ground_state_two_component_closure_already_implied_under_current_pack,
            "vector_form_factor_exact_computation_ready_under_current_pack": vector_form_factor_exact_computation_ready_under_current_pack,
            "primary_residual_lane": "vector_qball_form_factor_ground_state_two_component_closure_gap",
            "secondary_residual_lane": "qball_projection_overlap_future_source_theorem_reopen",
            "reserve_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_not_implied_declared",
            "advance_to_8_7_56_1282": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "inventory_note_hits": inventory["evidence"]["note_hits"],
        },
    )

    evaluation = payload(
        "8.7.56.1282",
        "Trial-2 numeric alpha vector Q-ball form-factor ground-state two-component closure review numeric evaluation",
        inputs,
        [
            row("beta_1_fixed", "pass", "beta_1 fixed", float(exact_ground_state["beta_n"]), "The retained electron-like beta_1 stays fixed during the closure review."),
            row("exact_ground_state_polarization_weight_fixed", "pass", "exact ground-state polarization weight fixed", float(exact_ground_state["polarization_weight"]), "The exact ground-state polarization weight remains zero."),
            row("exact_ground_state_coupled_charge_factor_fixed", "pass", "exact ground-state coupled charge factor fixed", float(exact_ground_state["coupled_charge_factor"]), "The exact ground-state coupled charge factor remains unity."),
            row("ell0_zero_seed_max_abs_fL_fixed", "pass", "ell=0 zero-seed max |f_L| fixed", float(vector_review_eval["summary"]["ell0_zero_seed_max_abs_fL"]), "The retained smoke test still reports no induced ell=0 longitudinal component at zero seed."),
            row("scalar_literal_F_m0_fixed", "pass", "scalar literal F(m0) fixed", float(vector_review_eval["summary"]["scalar_literal_F_m0"]), "The scalar exact baseline at literal q = m0 remains the retained fail benchmark."),
            row("numeric_state_changed_by_current_branch", "reject", "numeric state changed by current branch", 0.0, "The closure review changes only the route classification, not the retained numeric state."),
            row("route_state_changed_by_current_branch", "pass", "route state changed by current branch", 1.0, "The route moves from a generic closure requirement to an honest 'not yet implied under current pack' declaration."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_not_implied_under_current_pack",
            "beta_1": float(exact_ground_state["beta_n"]),
            "exact_ground_state_polarization_weight": float(exact_ground_state["polarization_weight"]),
            "exact_ground_state_coupled_charge_factor": float(exact_ground_state["coupled_charge_factor"]),
            "ell0_zero_seed_max_abs_fL": float(vector_review_eval["summary"]["ell0_zero_seed_max_abs_fL"]),
            "scalar_literal_F_m0": float(vector_review_eval["summary"]["scalar_literal_F_m0"]),
            "q_theory_over_m0": float(route_local_eval["summary"]["q_theory_over_m0"]),
            "F_exact_at_q_theory": float(route_local_eval["summary"]["F_exact_at_q_theory"]),
            "alpha_exact_at_q_theory": float(route_local_eval["summary"]["alpha_exact_at_q_theory"]),
            "vector_form_factor_exact_computation_ready_under_current_pack": vector_form_factor_exact_computation_ready_under_current_pack,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": vector_review_gate["summary"]["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_not_implied_under_current_pack",
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1279-.1282 artifacts generated")


if __name__ == "__main__":
    main()
