#!/usr/bin/env python3
"""Generate 8.7.56.1435-.1438 unified-closure Case C final summary route artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
EXPERT_SHARE = ROOT / "doc" / "quantum" / "38_trial2_numeric_alpha_vector_qball_exploratory_expert_share.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

UNIFIED_PLAN = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_unified_closure_plan_20260327.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

CLOSEOUT_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_closeout_sync_"
    "source_inventory_metrics.json"
)
CLOSEOUT_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_closeout_sync_"
    "audit_metrics.json"
)
CLOSEOUT_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_closeout_sync_"
    "declaration_gate_metrics.json"
)
CLOSEOUT_SHARE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_closeout_sync_"
    "share_pack_sync_metrics.json"
)
PHASE1_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "numeric_evaluation_metrics.json"
)
PHASE2_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_lambda_rot_form_factor_correction_"
    "numeric_evaluation_metrics.json"
)
PHASE3_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_blind_vector_observable_gate_"
    "numeric_evaluation_metrics.json"
)

SCRIPT_1431 = ROOT / "scripts" / "quantum" / "t2a_1431.py"

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_case_c_closeout_sync_completed"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_case_c_final_summary_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_registry"
NEXT_ROUTE = "8.7.56.1439"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_route"


# Function: return the current UTC timestamp string.
def now_iso() -> str:
    """Return the current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


# Function: fail fast when a required path is missing.

def require(path: Path) -> None:
    """Fail fast when a required path is missing."""
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


# Function: convert one path into repo-relative display form when possible.

def display_path(path: Path) -> str:
    """Convert one path into repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching line for one substring.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring."""
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


# Function: write one JSON payload and CSV rows table.

def write_artifact(kind: str, data: dict) -> None:
    """Write one JSON payload and CSV rows table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{STEM}_{kind}_metrics.json"
    csv_path = PUBLIC_OUT / f"{STEM}_{kind}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build one standard payload object.

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
    """Build one standard payload object."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: build one concise final-summary wording string from frozen metrics.

def build_final_summary_text(phase1_eval: dict, phase2_eval: dict, phase3_eval: dict) -> str:
    """Build one concise final-summary wording string from frozen metrics."""
    return (
        "v2.0 closes as Case C honest partial: the exact coupled ell=0 solver opened a nonzero regular f_L branch, "
        "but the lambda_rot correction remained a secondary-lane no-go and the blind vector observable failed at fixed "
        f"q_theory with F(q_theory)={phase3_eval['blind_F_at_q_theory']} and alpha(q_theory)={phase3_eval['blind_alpha_at_q_theory']}. "
        "The retained scalar baseline stays strong at "
        f"F_exact(q_theory)={phase1_eval['F_exact_at_q_theory_scalar']} and alpha_exact(q_theory)={phase1_eval['alpha_exact_at_q_theory_scalar']}, "
        "while physical_reject_required remains false."
    )


# Function: refresh one existing handoff bundle with the current synced files.

def refresh_handoff_bundle(bundle_dir: Path, bundle_zip: Path, files_to_sync: list[Path]) -> dict:
    """Refresh one existing handoff bundle with the current synced files."""
    copied_files: list[Path] = []
    for source in files_to_sync:
        target_path = bundle_dir / source.name
        shutil.copy2(source, target_path)
        copied_files.append(source)

    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(bundle_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(bundle_dir))

    with zipfile.ZipFile(bundle_zip, "r") as archive:
        zip_file_count = len(archive.namelist())

    return {
        "copied_files": copied_files,
        "copied_count": len(copied_files),
        "staging_file_count": len(list(bundle_dir.iterdir())),
        "zip_file_count": zip_file_count,
    }


# Function: execute the unified-closure Case C final summary route.

def main() -> None:
    """Execute the unified-closure Case C final summary route."""
    for path in (
        STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, PRIMARY_SOURCES, CURRENT_PROBLEM, CURRENT_STATUS,
        EXPERT_SHARE, UNIFIED_ROADMAP, PART1, PART3A, PART5, UNIFIED_PLAN, NEXT_STEPS, CLOSEOUT_INV, CLOSEOUT_AUDIT,
        CLOSEOUT_GATE, CLOSEOUT_SHARE, PHASE1_EVAL, PHASE2_EVAL, PHASE3_EVAL, SCRIPT_1431, Path(__file__)
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    expert_share_text = read_text(EXPERT_SHARE)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    unified_plan_text = read_text(UNIFIED_PLAN)
    next_steps_text = read_text(NEXT_STEPS)

    closeout_inv = read_json(CLOSEOUT_INV)["summary"]
    closeout_audit = read_json(CLOSEOUT_AUDIT)["summary"]
    closeout_gate = read_json(CLOSEOUT_GATE)["summary"]
    closeout_share = read_json(CLOSEOUT_SHARE)["summary"]
    phase1_eval = read_json(PHASE1_EVAL)["summary"]
    phase2_eval = read_json(PHASE2_EVAL)["summary"]
    phase3_eval = read_json(PHASE3_EVAL)["summary"]

    bundle_zip = ROOT / closeout_share["share_pack_bundle_zip"]
    bundle_dir = ROOT / closeout_share["share_pack_bundle_dir"]
    require(bundle_zip)
    require(bundle_dir)
    refreshed_bundle = refresh_handoff_bundle(
        bundle_dir,
        bundle_zip,
        [
            STATUS,
            ROADMAP,
            AI_CONTEXT,
            WORK_HISTORY_RECENT,
            PRIMARY_SOURCES,
            CURRENT_PROBLEM,
            CURRENT_STATUS,
            EXPERT_SHARE,
            UNIFIED_ROADMAP,
            PART1,
            PART3A,
            PART5,
            UNIFIED_PLAN,
            NEXT_STEPS,
            CLOSEOUT_INV,
            CLOSEOUT_AUDIT,
            CLOSEOUT_GATE,
            CLOSEOUT_SHARE,
            PHASE1_EVAL,
            PHASE2_EVAL,
            PHASE3_EVAL,
            SCRIPT_1431,
            Path(__file__),
        ],
    )

    summary_hits = [
        hit(status_text, "Case C final summary route"),
        hit(roadmap_text, "`8.7.56.1435-.1438`"),
        hit(current_problem_text, "Case C honest partial"),
        hit(current_status_text, "Case C honest partial"),
        hit(expert_share_text, "Case C honest partial"),
        hit(unified_roadmap_text, "Case C final summary route"),
        hit(part3a_text, "Case C final-summary next"),
        hit(part5_text, "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_route"),
        hit(unified_plan_text, "Case C"),
    ]
    inventory_ready = all(item is not None for item in summary_hits)
    closeout_metrics_available = bool(
        closeout_inv["inventory_ready"]
        and closeout_audit["case_c_honest_partial_wording_honest"]
        and closeout_gate["case_c_closeout_sync_ready"]
        and closeout_share["case_c_share_pack_sync_complete"]
    )
    share_pack_bundle_available = bool(bundle_zip.exists() and bundle_dir.exists())
    prior_route_ready = bool(closeout_gate["selected_next_generation_route"] == "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_route")

    final_summary_text = build_final_summary_text(phase1_eval, phase2_eval, phase3_eval)
    final_summary_word_count = len(final_summary_text.split())
    final_summary_concise = final_summary_word_count <= 80

    case_c_retained = bool(closeout_audit["case_c_honest_partial_wording_honest"])
    nonzero_fl_branch_retained = bool(closeout_audit["nonzero_fl_branch_retained"])
    phase2_no_go_retained = bool(closeout_audit["phase2_secondary_lane_no_go_retained"])
    phase3_no_go_retained = bool(closeout_audit["phase3_blind_observable_no_go_retained"])
    retained_scalar_candidate_retained = bool(closeout_audit["retained_scalar_strong_candidate_retained"])
    no_overclaim_exact_vector_charge = bool(closeout_audit["exact_vector_charge_derivation_not_overclaimed"])
    physical_reject_not_selected = not bool(closeout_gate["physical_reject_required"])

    wording_honest = all(
        [
            final_summary_concise,
            case_c_retained,
            nonzero_fl_branch_retained,
            phase2_no_go_retained,
            phase3_no_go_retained,
            retained_scalar_candidate_retained,
            no_overclaim_exact_vector_charge,
            physical_reject_not_selected,
        ]
    )
    final_summary_ready = bool(inventory_ready and closeout_metrics_available and share_pack_bundle_available and prior_route_ready and wording_honest)

    common_inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "primary_sources": display_path(PRIMARY_SOURCES),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "expert_share_note": display_path(EXPERT_SHARE),
            "unified_closure_roadmap_note": display_path(UNIFIED_ROADMAP),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "unified_plan_note": display_path(UNIFIED_PLAN),
            "next_steps_note": display_path(NEXT_STEPS),
        },
        "source_metrics": {
            "closeout_inventory": display_path(CLOSEOUT_INV),
            "closeout_audit": display_path(CLOSEOUT_AUDIT),
            "closeout_gate": display_path(CLOSEOUT_GATE),
            "closeout_share_pack": display_path(CLOSEOUT_SHARE),
            "phase1_eval": display_path(PHASE1_EVAL),
            "phase2_eval": display_path(PHASE2_EVAL),
            "phase3_eval": display_path(PHASE3_EVAL),
        },
        "scripts": {
            "closeout_sync": display_path(SCRIPT_1431),
            "final_summary_route": display_path(Path(__file__)),
        },
        "constants": {
            "prior_classification": PRIOR_CLASS,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "bundle_zip": display_path(bundle_zip),
        },
    }

    inventory_payload = payload(
        "8.7.56.1435",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Case C final summary inventory",
        common_inputs,
        [
            row(
                "inventory_complete",
                "pass" if inventory_ready else "reject",
                "Case C final summary inventory complete",
                1 if inventory_ready else 0,
                "The final summary inventory is ready only if the closeout metrics, share-pack bundle, current notes, and paper wording coexist in one pack.",
            ),
            row(
                "closeout_metrics_available",
                "pass" if closeout_metrics_available else "reject",
                "Case C closeout metrics available",
                1 if closeout_metrics_available else 0,
                "The final summary route starts only after the closeout inventory, wording audit, declaration gate, and share-pack sync are all already fixed.",
            ),
            row(
                "share_pack_bundle_available",
                "pass" if share_pack_bundle_available else "reject",
                "Case C share-pack bundle available for final summary handoff",
                1 if share_pack_bundle_available else 0,
                "The final summary route reuses the synchronized Case C share-pack as the canonical handoff artifact.",
            ),
            row(
                "prior_route_selected",
                "pass" if prior_route_ready else "reject",
                "Case C final summary route already selected by closeout gate",
                1 if prior_route_ready else 0,
                "The final summary route is valid only if the prior closeout gate explicitly selected it.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "inventory_ready": inventory_ready,
            "closeout_metrics_available": closeout_metrics_available,
            "share_pack_bundle_available": share_pack_bundle_available,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "share_pack_bundle_refresh_count": refreshed_bundle["copied_count"],
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_inventory_fixed",
            "advance_to_8_7_56_1436": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "summary_hits": summary_hits,
            "final_summary_text": final_summary_text,
            "final_summary_word_count": final_summary_word_count,
        },
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1436",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Case C final summary wording audit",
        common_inputs,
        [
            row(
                "case_c_final_summary_wording_honest",
                "pass" if wording_honest else "reject",
                "Case C final summary wording honest",
                1 if wording_honest else 0,
                "The final summary wording is honest only if it keeps Case C honest partial, the Phase 2 and Phase 3 no-go limits, the retained scalar strong candidate, and physical_reject_required=false without overclaiming vector closure.",
            ),
            row(
                "case_c_final_summary_wording_concise",
                "pass" if final_summary_concise else "reject",
                "Case C final summary wording concise",
                1 if final_summary_concise else 0,
                "The final summary route should compress the Case C disposition into a short final-summary paragraph rather than reopening theorem-side detail.",
            ),
            row(
                "case_c_honest_partial_retained",
                "pass" if case_c_retained else "reject",
                "Case C honest partial retained in final summary",
                1 if case_c_retained else 0,
                "The final summary must keep the route-local Case C honest-partial reading intact.",
            ),
            row(
                "nonzero_fl_branch_retained",
                "pass" if nonzero_fl_branch_retained else "reject",
                "nonzero f_L branch retained in final summary",
                1 if nonzero_fl_branch_retained else 0,
                "The final summary must still say that the exact coupled solver opened a nonzero regular f_L branch.",
            ),
            row(
                "phase2_secondary_lane_no_go_retained",
                "pass" if phase2_no_go_retained else "reject",
                "Phase 2 secondary-lane no-go retained in final summary",
                1 if phase2_no_go_retained else 0,
                "The final summary must preserve that the lambda_rot correction did not close the route.",
            ),
            row(
                "phase3_blind_observable_no_go_retained",
                "pass" if phase3_no_go_retained else "reject",
                "Phase 3 blind-observable no-go retained in final summary",
                1 if phase3_no_go_retained else 0,
                "The final summary must preserve that the blind vector observable route failed at fixed q_theory.",
            ),
            row(
                "retained_scalar_strong_candidate_retained",
                "pass" if retained_scalar_candidate_retained else "reject",
                "retained scalar strong candidate retained in final summary",
                1 if retained_scalar_candidate_retained else 0,
                "The final summary must keep the strong scalar baseline visible so Case C does not read like a full reject.",
            ),
            row(
                "exact_vector_charge_derivation_not_overclaimed",
                "pass" if no_overclaim_exact_vector_charge else "reject",
                "exact vector-charge derivation not overclaimed in final summary",
                1 if no_overclaim_exact_vector_charge else 0,
                "The final summary must not promote an exact vector-charge derivation while source theorem and observable dictionary remain unresolved.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected in final summary",
                1 if physical_reject_not_selected else 0,
                "The final summary preserves the route-local reading and therefore keeps physical_reject_required=false.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "case_c_final_summary_wording_honest": wording_honest,
            "case_c_final_summary_wording_concise": final_summary_concise,
            "case_c_honest_partial_retained": case_c_retained,
            "nonzero_fl_branch_retained": nonzero_fl_branch_retained,
            "phase2_secondary_lane_no_go_retained": phase2_no_go_retained,
            "phase3_blind_observable_no_go_retained": phase3_no_go_retained,
            "retained_scalar_strong_candidate_retained": retained_scalar_candidate_retained,
            "exact_vector_charge_derivation_not_overclaimed": no_overclaim_exact_vector_charge,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_audited",
            "advance_to_8_7_56_1437": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "final_summary_text": final_summary_text,
            "final_summary_word_count": final_summary_word_count,
            "phase1_summary": phase1_eval,
            "phase2_summary": phase2_eval,
            "phase3_summary": phase3_eval,
        },
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1437",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Case C final summary declaration gate",
        common_inputs,
        [
            row(
                "case_c_final_summary_ready",
                "pass" if final_summary_ready else "reject",
                "Case C final summary ready",
                1 if final_summary_ready else 0,
                "The final summary can be declared only after the inventory, wording audit, and closeout sync all agree.",
            ),
            row(
                "v2_case_c_final_summary_fixed",
                "pass" if final_summary_ready else "reject",
                "v2.0 Case C final summary fixed",
                1 if final_summary_ready else 0,
                "The declaration gate fixes the concise v2.0 final summary wording for Case C without reopening theorem-side branches.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected after final summary",
                1 if physical_reject_not_selected else 0,
                "The final summary keeps the route-local reading and does not force a physical reject.",
            ),
            row(
                "next_route_selected",
                "pass" if final_summary_ready else "reject",
                "future reopen-trigger registry selected as next route",
                1 if final_summary_ready else 0,
                "After the final summary is fixed, the next work is a reopen-trigger registry rather than another closeout restatement.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "case_c_final_summary_ready": final_summary_ready,
            "v2_0_final_disposition_case": "Case C",
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_gate_closed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"final_summary_text": final_summary_text, "bundle_zip": display_path(bundle_zip)},
    )
    write_artifact("declaration_gate", gate_payload)

    handoff_payload = payload(
        "8.7.56.1438",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure Case C final summary handoff sync",
        common_inputs,
        [
            row(
                "final_summary_handoff_sync_complete",
                "pass" if final_summary_ready else "reject",
                "Case C final summary handoff sync complete",
                1 if final_summary_ready else 0,
                "Handoff sync is complete only when the final summary route is ready and the canonical share-pack bundle remains available.",
            ),
            row(
                "share_pack_bundle_retained",
                "pass" if share_pack_bundle_available else "reject",
                "Case C share-pack bundle retained for handoff",
                1 if share_pack_bundle_available else 0,
                "The final summary handoff reuses the existing closeout share-pack bundle instead of regenerating a new theorem-side package.",
            ),
            row(
                "share_pack_bundle_refreshed",
                "pass" if refreshed_bundle["copied_count"] > 0 else "reject",
                "Case C share-pack bundle refreshed with synced final-summary files",
                refreshed_bundle["copied_count"],
                "The final summary handoff refreshes the canonical bundle in place so the synced notes and metrics match the active route state.",
            ),
            row(
                "expert_share_note_sync_ready",
                "pass" if inventory_ready else "reject",
                "expert-share note sync ready",
                1 if inventory_ready else 0,
                "The expert-share note can be synchronized because the final summary route is already grounded in the canonical notes and paper wording.",
            ),
            row(
                "future_reopen_trigger_registry_selected",
                "pass" if final_summary_ready else "reject",
                "future reopen-trigger registry selected",
                1 if final_summary_ready else 0,
                "After the final summary handoff is frozen, the route advances to a registry of future reopen triggers.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "case_c_final_summary_handoff_sync_complete": final_summary_ready,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "share_pack_bundle_refresh_count": refreshed_bundle["copied_count"],
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_handoff_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "final_summary_text": final_summary_text,
            "hints_for_next_registry": [
                "exact_action_level_ell0_operator_reopen",
                "future_source_theorem_reopen",
                "observable_dictionary_exact_charge_current_bridge",
            ],
            "refreshed_bundle_copied_files": [display_path(path) for path in refreshed_bundle["copied_files"]],
            "refreshed_bundle_staging_file_count": refreshed_bundle["staging_file_count"],
            "refreshed_bundle_zip_file_count": refreshed_bundle["zip_file_count"],
            "bundle_zip": display_path(bundle_zip),
            "bundle_dir": display_path(bundle_dir),
        },
    )
    write_artifact("handoff_sync", handoff_payload)

    print("[done] 8.7.56.1435-.1438 artifacts generated")
    print(f"[done] final_summary_word_count={final_summary_word_count}")
    print(f"[done] bundle_zip={bundle_zip}")


if __name__ == "__main__":
    main()
