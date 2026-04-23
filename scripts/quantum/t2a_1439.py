#!/usr/bin/env python3
"""Generate 8.7.56.1439-.1442 unified-closure future reopen-trigger registry artifacts."""

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

FINAL_SUMMARY_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_route_"
    "source_inventory_metrics.json"
)
FINAL_SUMMARY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_route_"
    "audit_metrics.json"
)
FINAL_SUMMARY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_route_"
    "declaration_gate_metrics.json"
)
FINAL_SUMMARY_HANDOFF = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_route_"
    "handoff_sync_metrics.json"
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

SCRIPT_1435 = ROOT / "scripts" / "quantum" / "t2a_1435.py"

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_case_c_final_summary_completed"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_future_reopen_trigger_registry_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_closeout_sync"
NEXT_ROUTE = "8.7.56.1443"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_registry"

PRIMARY_TRIGGER = "exact_action_level_ell0_operator_reopen"
SECONDARY_TRIGGER = "future_source_theorem_reopen"
RESERVE_TRIGGER = "observable_dictionary_exact_charge_current_bridge"


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

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
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


# Function: execute the unified-closure future reopen-trigger registry branch.

def main() -> None:
    """Execute the unified-closure future reopen-trigger registry branch."""
    for path in (
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
        FINAL_SUMMARY_INV,
        FINAL_SUMMARY_AUDIT,
        FINAL_SUMMARY_GATE,
        FINAL_SUMMARY_HANDOFF,
        PHASE1_EVAL,
        PHASE2_EVAL,
        PHASE3_EVAL,
        SCRIPT_1435,
        Path(__file__),
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

    final_summary_inv = read_json(FINAL_SUMMARY_INV)["summary"]
    final_summary_audit = read_json(FINAL_SUMMARY_AUDIT)["summary"]
    final_summary_gate = read_json(FINAL_SUMMARY_GATE)["summary"]
    final_summary_handoff = read_json(FINAL_SUMMARY_HANDOFF)["summary"]
    phase1_eval = read_json(PHASE1_EVAL)["summary"]
    phase2_eval = read_json(PHASE2_EVAL)["summary"]
    phase3_eval = read_json(PHASE3_EVAL)["summary"]

    bundle_zip = ROOT / final_summary_handoff["share_pack_bundle_zip"]
    bundle_dir = bundle_zip.with_suffix("")
    require(bundle_zip)
    require(bundle_dir)

    reopen_hits_required = [
        hit(status_text, "future reopen-trigger"),
        hit(roadmap_text, "`8.7.56.1439-.1442`"),
        hit(current_problem_text, PRIMARY_TRIGGER),
        hit(current_problem_text, SECONDARY_TRIGGER),
        hit(current_problem_text, RESERVE_TRIGGER),
        hit(current_status_text, "future reopen trigger ordering"),
        hit(expert_share_text, "first reopen surface"),
        hit(unified_roadmap_text, "future reopen-trigger registry"),
        hit(part3a_text, "future reopen-trigger registry next"),
        hit(part5_text, "future_reopen_trigger_registry"),
    ]
    reopen_hits_optional = [
        hit(unified_plan_text, "Case C"),
        hit(next_steps_text, "source theorem"),
    ]
    inventory_ready = all(item is not None for item in reopen_hits_required)

    final_summary_ready = bool(
        final_summary_inv["inventory_ready"]
        and final_summary_audit["case_c_final_summary_wording_honest"]
        and final_summary_gate["case_c_final_summary_ready"]
        and final_summary_handoff["case_c_final_summary_handoff_sync_complete"]
    )
    reopen_surfaces_available = bool(
        phase1_eval["nonzero_regular_branch_detected"]
        and not phase1_eval["phase1_close_within_one_percent"]
        and phase2_eval["phase2_secondary_lane_no_go"]
        and phase3_eval["case_c_selected"]
    )
    share_pack_bundle_available = bool(bundle_zip.exists() and bundle_dir.exists())

    primary_ready = bool(
        phase1_eval["nonzero_regular_branch_detected"]
        and not phase1_eval["phase1_close_within_one_percent"]
        and not phase1_eval["primary_lane_no_go"]
    )
    secondary_ready = bool(
        phase2_eval["phase2_secondary_lane_no_go"]
        and phase2_eval["phase3_required"]
        and not phase2_eval["physical_reject_required"]
    )
    reserve_ready = bool(
        phase3_eval["blind_observable_gate_pass"] is False
        and phase3_eval["case_c_selected"]
        and not phase3_eval["physical_reject_required"]
    )
    ordering_honest = all(
        [
            primary_ready,
            secondary_ready,
            reserve_ready,
            final_summary_audit["case_c_honest_partial_retained"],
            final_summary_audit["retained_scalar_strong_candidate_retained"],
            not final_summary_audit["physical_reject_required"],
        ]
    )
    registry_ready = all([inventory_ready, final_summary_ready, reopen_surfaces_available, ordering_honest])

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
            "final_summary_inventory": display_path(FINAL_SUMMARY_INV),
            "final_summary_audit": display_path(FINAL_SUMMARY_AUDIT),
            "final_summary_gate": display_path(FINAL_SUMMARY_GATE),
            "final_summary_handoff": display_path(FINAL_SUMMARY_HANDOFF),
            "phase1_eval": display_path(PHASE1_EVAL),
            "phase2_eval": display_path(PHASE2_EVAL),
            "phase3_eval": display_path(PHASE3_EVAL),
        },
        "scripts": {
            "case_c_final_summary_route": display_path(SCRIPT_1435),
            "future_reopen_trigger_registry": display_path(Path(__file__)),
        },
        "constants": {
            "prior_classification": PRIOR_CLASS,
            "primary_trigger": PRIMARY_TRIGGER,
            "secondary_trigger": SECONDARY_TRIGGER,
            "reserve_trigger": RESERVE_TRIGGER,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "bundle_zip": display_path(bundle_zip),
        },
    }

    inventory_payload = payload(
        "8.7.56.1439",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger inventory",
        common_inputs,
        [
            row(
                "inventory_ready",
                "pass" if inventory_ready else "reject",
                "future reopen-trigger inventory ready",
                1 if inventory_ready else 0,
                "The registry inventory is ready only if the final summary metrics, reopen surfaces, paper wording, and expert-share/current notes coexist in one pack.",
            ),
            row(
                "final_summary_metrics_available",
                "pass" if final_summary_ready else "reject",
                "Case C final summary metrics available",
                1 if final_summary_ready else 0,
                "The reopen-trigger registry starts only after the final summary route is already fixed and synced.",
            ),
            row(
                "reopen_surfaces_available",
                "pass" if reopen_surfaces_available else "reject",
                "reopen surfaces available after Case C closeout",
                1 if reopen_surfaces_available else 0,
                "Primary, secondary, and reserve reopen surfaces are available only if the exact coupled branch, lambda_rot no-go, and blind-observable Case C state remain visible together.",
            ),
            row(
                "share_pack_bundle_available",
                "pass" if share_pack_bundle_available else "reject",
                "share-pack bundle available for reopen handoff",
                1 if share_pack_bundle_available else 0,
                "The reopen-trigger registry reuses the canonical Case C handoff bundle as the expert-share baseline.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "inventory_ready": inventory_ready,
            "case_c_final_summary_metrics_available": final_summary_ready,
            "reopen_surfaces_available": reopen_surfaces_available,
            "share_pack_bundle_available": share_pack_bundle_available,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_inventory_fixed",
            "advance_to_8_7_56_1440": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "reopen_hits_required": reopen_hits_required,
            "reopen_hits_optional": reopen_hits_optional,
            "phase1_best_alpha": phase1_eval["phase1_best_alpha_candidate"]["alpha_at_q_theory"],
            "phase2_best_additive_alpha": phase2_eval["best_naive_add_3sigma_alpha"],
            "phase3_signed_crossing_over_m0": phase3_eval["signed_target_crossing_over_m0"],
        },
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1440",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger ordering audit",
        common_inputs,
        [
            row(
                "primary_trigger_exact_action_level_ell0_operator_reopen",
                "pass" if primary_ready else "reject",
                "exact-action-level ell=0 operator reopen retained as primary trigger",
                1 if primary_ready else 0,
                "The primary reopen trigger is honest only if the exact coupled solver opened a nonzero regular branch without closing within one percent and without forcing a primary-lane no-go.",
            ),
            row(
                "secondary_trigger_future_source_theorem_reopen",
                "pass" if secondary_ready else "reject",
                "future source-theorem reopen retained as secondary trigger",
                1 if secondary_ready else 0,
                "The secondary reopen trigger remains honest only if lambda_rot stayed a no-go while still preserving a non-reject path.",
            ),
            row(
                "reserve_trigger_observable_dictionary_exact_charge_current_bridge",
                "pass" if reserve_ready else "reject",
                "observable dictionary exact charge-current bridge retained as reserve trigger",
                1 if reserve_ready else 0,
                "The reserve reopen trigger remains honest only if the blind observable route closed only as Case C partial and not as a physical reject.",
            ),
            row(
                "case_c_final_summary_retained",
                "pass" if final_summary_audit["case_c_honest_partial_retained"] else "reject",
                "Case C final summary retained while ordering reopen triggers",
                1 if final_summary_audit["case_c_honest_partial_retained"] else 0,
                "The registry must preserve the Case C honest-partial wording rather than rewrite the disposition.",
            ),
            row(
                "retained_scalar_strong_candidate_retained",
                "pass" if final_summary_audit["retained_scalar_strong_candidate_retained"] else "reject",
                "retained scalar strong candidate kept visible in reopen ordering",
                1 if final_summary_audit["retained_scalar_strong_candidate_retained"] else 0,
                "The reopen ordering remains honest only if it keeps the strong scalar baseline visible.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if not final_summary_audit["physical_reject_required"] else "reject",
                "physical reject not selected in reopen ordering",
                1 if not final_summary_audit["physical_reject_required"] else 0,
                "The reopen registry is only a future trigger ordering and therefore must keep physical_reject_required=false.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_ordering_honest": ordering_honest,
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "case_c_honest_partial_retained": final_summary_audit["case_c_honest_partial_retained"],
            "retained_scalar_strong_candidate_retained": final_summary_audit["retained_scalar_strong_candidate_retained"],
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_ordering_audited",
            "advance_to_8_7_56_1441": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "primary_trigger_evidence": phase1_eval["phase1_best_alpha_candidate"],
            "secondary_trigger_evidence": {
                "phase2_secondary_lane_no_go": phase2_eval["phase2_secondary_lane_no_go"],
                "required_alpha_multiplier": phase2_eval["required_alpha_multiplier"],
                "best_naive_add_3sigma_alpha": phase2_eval["best_naive_add_3sigma_alpha"],
            },
            "reserve_trigger_evidence": {
                "blind_F_at_q_theory": phase3_eval["blind_F_at_q_theory"],
                "blind_alpha_at_q_theory": phase3_eval["blind_alpha_at_q_theory"],
                "signed_target_crossing_over_m0": phase3_eval["signed_target_crossing_over_m0"],
            },
        },
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1441",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger declaration gate",
        common_inputs,
        [
            row(
                "future_reopen_trigger_registry_ready",
                "pass" if registry_ready else "reject",
                "future reopen-trigger registry ready",
                1 if registry_ready else 0,
                "The registry is ready only when the final summary is fixed, reopen surfaces are present, and the ordering audit is honest.",
            ),
            row(
                "future_reopen_trigger_ordering_fixed",
                "pass" if registry_ready else "reject",
                "future reopen-trigger ordering fixed",
                1 if registry_ready else 0,
                "The declaration gate freezes the primary / secondary / reserve reopen-trigger ordering after Case C closeout.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if not final_summary_audit["physical_reject_required"] else "reject",
                "physical reject not selected after future reopen-trigger registry",
                1 if not final_summary_audit["physical_reject_required"] else 0,
                "The reopen-trigger registry retains a future-looking ordering rather than escalating to physical reject.",
            ),
            row(
                "next_route_selected",
                "pass" if registry_ready else "reject",
                "future reopen-trigger closeout sync selected as next route",
                1 if registry_ready else 0,
                "After the registry is frozen, the next work is to sync and close out the reopen-trigger state for expert handoff.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_registry_ready": registry_ready,
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_gate_closed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "retained_scalar_alpha_exact_at_q_theory": phase1_eval["alpha_exact_at_q_theory_scalar"],
            "phase1_best_vector_alpha": phase1_eval["phase1_best_alpha_candidate"]["alpha_at_q_theory"],
            "phase3_blind_signed_crossing_over_m0": phase3_eval["signed_target_crossing_over_m0"],
        },
    )
    write_artifact("declaration_gate", gate_payload)

    refresh_candidates = [
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
        FINAL_SUMMARY_INV,
        FINAL_SUMMARY_AUDIT,
        FINAL_SUMMARY_GATE,
        FINAL_SUMMARY_HANDOFF,
        PHASE1_EVAL,
        PHASE2_EVAL,
        PHASE3_EVAL,
        SCRIPT_1435,
        Path(__file__),
        PUBLIC_OUT / f"{STEM}_source_inventory_metrics.json",
        PUBLIC_OUT / f"{STEM}_audit_metrics.json",
        PUBLIC_OUT / f"{STEM}_declaration_gate_metrics.json",
    ]
    refreshed_bundle = refresh_handoff_bundle(bundle_dir, bundle_zip, refresh_candidates)

    handoff_payload = payload(
        "8.7.56.1442",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger expert handoff sync",
        common_inputs,
        [
            row(
                "future_reopen_trigger_handoff_sync_complete",
                "pass" if registry_ready else "reject",
                "future reopen-trigger handoff sync complete",
                1 if registry_ready else 0,
                "Handoff sync is complete only when the reopen-trigger registry is ready and the canonical bundle has been refreshed with the registry artifacts.",
            ),
            row(
                "share_pack_bundle_retained",
                "pass" if share_pack_bundle_available else "reject",
                "canonical share-pack bundle retained for reopen handoff",
                1 if share_pack_bundle_available else 0,
                "The registry handoff reuses the Case C canonical bundle rather than opening a new theorem-side package.",
            ),
            row(
                "share_pack_bundle_refreshed",
                "pass" if refreshed_bundle["copied_count"] > 0 else "reject",
                "canonical share-pack bundle refreshed with reopen-trigger files",
                refreshed_bundle["copied_count"],
                "The registry handoff refreshes the canonical bundle in place so expert readers see the active reopen-trigger ordering.",
            ),
            row(
                "expert_handoff_note_sync_ready",
                "pass" if inventory_ready else "reject",
                "expert handoff notes sync ready",
                1 if inventory_ready else 0,
                "The expert handoff is ready because the final summary, reopen ordering, and current notes now point to the same future trigger registry.",
            ),
            row(
                "future_reopen_trigger_closeout_sync_selected",
                "pass" if registry_ready else "reject",
                "future reopen-trigger closeout sync selected",
                1 if registry_ready else 0,
                "After the registry handoff is synced, the next work is a closeout sync for the frozen reopen ordering.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_handoff_sync_complete": registry_ready,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "share_pack_bundle_refresh_count": refreshed_bundle["copied_count"],
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "refreshed_bundle_copied_files": [display_path(path) for path in refreshed_bundle["copied_files"]],
            "refreshed_bundle_staging_file_count": refreshed_bundle["staging_file_count"],
            "refreshed_bundle_zip_file_count": refreshed_bundle["zip_file_count"],
            "bundle_zip": display_path(bundle_zip),
            "bundle_dir": display_path(bundle_dir),
            "reopen_trigger_hints": [PRIMARY_TRIGGER, SECONDARY_TRIGGER, RESERVE_TRIGGER],
        },
    )
    write_artifact("expert_handoff_sync", handoff_payload)

    print("[done] 8.7.56.1439-.1442 artifacts generated")
    print(f"[done] bundle_zip={bundle_zip}")
    print(f"[done] primary={PRIMARY_TRIGGER}")


if __name__ == "__main__":
    main()
