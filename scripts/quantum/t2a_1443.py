#!/usr/bin/env python3
"""Generate 8.7.56.1443-.1446 unified-closure future reopen-trigger closeout sync artifacts."""

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

REOPEN_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_registry_"
    "source_inventory_metrics.json"
)
REOPEN_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_registry_"
    "audit_metrics.json"
)
REOPEN_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_registry_"
    "declaration_gate_metrics.json"
)
REOPEN_HANDOFF = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_registry_"
    "expert_handoff_sync_metrics.json"
)
FINAL_SUMMARY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_case_c_final_summary_route_"
    "audit_metrics.json"
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

SCRIPT_1439 = ROOT / "scripts" / "quantum" / "t2a_1439.py"

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_future_reopen_trigger_registry_completed"
BRANCH_CLASS = "vector_qball_form_factor_unified_closure_future_reopen_trigger_closeout_sync_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_final_summary_route"
NEXT_ROUTE = "8.7.56.1447"
STEM = "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_closeout_sync"

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


# Function: execute the unified-closure future reopen-trigger closeout sync branch.

def main() -> None:
    """Execute the unified-closure future reopen-trigger closeout sync branch."""
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
        REOPEN_INV,
        REOPEN_AUDIT,
        REOPEN_GATE,
        REOPEN_HANDOFF,
        FINAL_SUMMARY_AUDIT,
        PHASE1_EVAL,
        PHASE2_EVAL,
        PHASE3_EVAL,
        SCRIPT_1439,
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

    reopen_inv = read_json(REOPEN_INV)["summary"]
    reopen_audit = read_json(REOPEN_AUDIT)["summary"]
    reopen_gate = read_json(REOPEN_GATE)["summary"]
    reopen_handoff = read_json(REOPEN_HANDOFF)["summary"]
    final_summary_audit = read_json(FINAL_SUMMARY_AUDIT)["summary"]
    phase1_eval = read_json(PHASE1_EVAL)["summary"]
    phase2_eval = read_json(PHASE2_EVAL)["summary"]
    phase3_eval = read_json(PHASE3_EVAL)["summary"]

    bundle_zip = ROOT / reopen_handoff["share_pack_bundle_zip"]
    bundle_dir = bundle_zip.with_suffix("")
    require(bundle_zip)
    require(bundle_dir)

    closeout_hits_required = [
        hit(status_text, "future reopen-trigger closeout"),
        hit(roadmap_text, "`8.7.56.1443-.1446`"),
        hit(current_problem_text, "future reopen-trigger closeout"),
        hit(current_status_text, "future reopen-trigger closeout"),
        hit(expert_share_text, "current official state"),
        hit(unified_roadmap_text, "future reopen-trigger closeout sync"),
        hit(part3a_text, "future reopen-trigger closeout-sync next"),
        hit(part5_text, "future_reopen_trigger_closeout_sync"),
    ]
    closeout_hits_optional = [
        hit(unified_plan_text, "Case C"),
        hit(next_steps_text, "source theorem"),
    ]
    inventory_ready = all(item is not None for item in closeout_hits_required)

    registry_ready = bool(
        reopen_inv["inventory_ready"]
        and reopen_audit["future_reopen_trigger_ordering_honest"]
        and reopen_gate["future_reopen_trigger_registry_ready"]
        and reopen_handoff["future_reopen_trigger_handoff_sync_complete"]
    )
    share_pack_bundle_available = bool(bundle_zip.exists() and bundle_dir.exists())
    case_c_honest_partial_retained = bool(final_summary_audit["case_c_honest_partial_retained"])
    retained_scalar_candidate_retained = bool(final_summary_audit["retained_scalar_strong_candidate_retained"])
    physical_reject_not_selected = not bool(final_summary_audit["physical_reject_required"])
    ordering_retained = bool(
        reopen_audit["primary_future_reopen_trigger"] == PRIMARY_TRIGGER
        and reopen_audit["secondary_future_reopen_trigger"] == SECONDARY_TRIGGER
        and reopen_audit["reserve_future_reopen_trigger"] == RESERVE_TRIGGER
    )
    closeout_wording_honest = all(
        [
            registry_ready,
            case_c_honest_partial_retained,
            retained_scalar_candidate_retained,
            physical_reject_not_selected,
            ordering_retained,
        ]
    )
    closeout_sync_ready = all([inventory_ready, registry_ready, share_pack_bundle_available, closeout_wording_honest])

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
            "reopen_inventory": display_path(REOPEN_INV),
            "reopen_audit": display_path(REOPEN_AUDIT),
            "reopen_gate": display_path(REOPEN_GATE),
            "reopen_handoff": display_path(REOPEN_HANDOFF),
            "final_summary_audit": display_path(FINAL_SUMMARY_AUDIT),
            "phase1_eval": display_path(PHASE1_EVAL),
            "phase2_eval": display_path(PHASE2_EVAL),
            "phase3_eval": display_path(PHASE3_EVAL),
        },
        "scripts": {
            "future_reopen_trigger_registry": display_path(SCRIPT_1439),
            "future_reopen_trigger_closeout_sync": display_path(Path(__file__)),
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
        "8.7.56.1443",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger closeout inventory",
        common_inputs,
        [
            row(
                "inventory_ready",
                "pass" if inventory_ready else "reject",
                "future reopen-trigger closeout inventory ready",
                1 if inventory_ready else 0,
                "The closeout inventory is ready only if the reopen-trigger registry metrics, canonical bundle, paper wording, and synced notes coexist in one pack.",
            ),
            row(
                "future_reopen_trigger_registry_available",
                "pass" if registry_ready else "reject",
                "future reopen-trigger registry available for closeout sync",
                1 if registry_ready else 0,
                "The closeout sync starts only after the reopen-trigger registry is already frozen and handed off.",
            ),
            row(
                "share_pack_bundle_available",
                "pass" if share_pack_bundle_available else "reject",
                "canonical share-pack bundle available for closeout sync",
                1 if share_pack_bundle_available else 0,
                "The closeout sync reuses the canonical expert-share bundle produced by the prior reopen-trigger registry.",
            ),
            row(
                "ordering_retained",
                "pass" if ordering_retained else "reject",
                "future reopen-trigger ordering retained for closeout sync",
                1 if ordering_retained else 0,
                "The closeout sync must keep the primary / secondary / reserve trigger ordering unchanged.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "inventory_ready": inventory_ready,
            "future_reopen_trigger_registry_available": registry_ready,
            "share_pack_bundle_available": share_pack_bundle_available,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_closeout_inventory_fixed",
            "advance_to_8_7_56_1444": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "closeout_hits_required": closeout_hits_required,
            "closeout_hits_optional": closeout_hits_optional,
            "phase1_best_vector_alpha": phase1_eval["phase1_best_alpha_candidate"]["alpha_at_q_theory"],
            "phase2_best_additive_alpha": phase2_eval["best_naive_add_3sigma_alpha"],
            "phase3_blind_alpha_at_q_theory": phase3_eval["blind_alpha_at_q_theory"],
        },
    )
    write_artifact("source_inventory", inventory_payload)

    audit_payload = payload(
        "8.7.56.1444",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger closeout wording audit",
        common_inputs,
        [
            row(
                "future_reopen_trigger_closeout_wording_honest",
                "pass" if closeout_wording_honest else "reject",
                "future reopen-trigger closeout wording honest",
                1 if closeout_wording_honest else 0,
                "The closeout sync wording is honest only if it preserves Case C honest partial, the retained scalar strong baseline, the frozen reopen ordering, and physical_reject_required=false.",
            ),
            row(
                "case_c_honest_partial_retained",
                "pass" if case_c_honest_partial_retained else "reject",
                "Case C honest partial retained in closeout wording",
                1 if case_c_honest_partial_retained else 0,
                "The closeout wording must still say that the route closes only as Case C honest partial.",
            ),
            row(
                "retained_scalar_strong_candidate_retained",
                "pass" if retained_scalar_candidate_retained else "reject",
                "retained scalar strong candidate kept visible in closeout wording",
                1 if retained_scalar_candidate_retained else 0,
                "The closeout wording remains honest only if it keeps the strong scalar baseline visible.",
            ),
            row(
                "future_reopen_ordering_retained",
                "pass" if ordering_retained else "reject",
                "future reopen ordering retained in closeout wording",
                1 if ordering_retained else 0,
                "The closeout wording must preserve the exact operator / future source theorem / observable dictionary ordering without reclassification.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected in closeout wording",
                1 if physical_reject_not_selected else 0,
                "The closeout wording is still a retained partial state and must keep physical_reject_required=false.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_closeout_wording_honest": closeout_wording_honest,
            "case_c_honest_partial_retained": case_c_honest_partial_retained,
            "retained_scalar_strong_candidate_retained": retained_scalar_candidate_retained,
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_closeout_wording_audited",
            "advance_to_8_7_56_1445": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "reopen_ordering_summary": {
                "primary": PRIMARY_TRIGGER,
                "secondary": SECONDARY_TRIGGER,
                "reserve": RESERVE_TRIGGER,
            },
            "retained_scalar_alpha_exact_at_q_theory": phase1_eval["alpha_exact_at_q_theory_scalar"],
            "best_phase1_vector_alpha": phase1_eval["phase1_best_alpha_candidate"]["alpha_at_q_theory"],
            "phase3_blind_signed_crossing_over_m0": phase3_eval["signed_target_crossing_over_m0"],
        },
    )
    write_artifact("audit", audit_payload)

    gate_payload = payload(
        "8.7.56.1445",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger closeout declaration gate",
        common_inputs,
        [
            row(
                "future_reopen_trigger_closeout_sync_ready",
                "pass" if closeout_sync_ready else "reject",
                "future reopen-trigger closeout sync ready",
                1 if closeout_sync_ready else 0,
                "The closeout sync is ready only after the registry is frozen, the wording audit passes, and the canonical bundle remains available.",
            ),
            row(
                "future_reopen_trigger_closeout_wording_fixed",
                "pass" if closeout_sync_ready else "reject",
                "future reopen-trigger closeout wording fixed",
                1 if closeout_sync_ready else 0,
                "The declaration gate freezes the closeout wording for the reopen-trigger ordering without reopening the computation lanes.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected after closeout sync",
                1 if physical_reject_not_selected else 0,
                "The closeout sync retains a future-reopen reading rather than escalating to physical reject.",
            ),
            row(
                "next_route_selected",
                "pass" if closeout_sync_ready else "reject",
                "future reopen-trigger final summary route selected as next route",
                1 if closeout_sync_ready else 0,
                "After the closeout sync is frozen, the next work is a concise final-summary route for the reopen-trigger state.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_closeout_sync_ready": closeout_sync_ready,
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_closeout_gate_closed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "retained_scalar_alpha_exact_at_q_theory": phase1_eval["alpha_exact_at_q_theory_scalar"],
            "best_phase1_vector_alpha": phase1_eval["phase1_best_alpha_candidate"]["alpha_at_q_theory"],
            "blind_vector_alpha_at_q_theory": phase3_eval["blind_alpha_at_q_theory"],
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
        REOPEN_INV,
        REOPEN_AUDIT,
        REOPEN_GATE,
        REOPEN_HANDOFF,
        FINAL_SUMMARY_AUDIT,
        PHASE1_EVAL,
        PHASE2_EVAL,
        PHASE3_EVAL,
        SCRIPT_1439,
        Path(__file__),
        PUBLIC_OUT / f"{STEM}_source_inventory_metrics.json",
        PUBLIC_OUT / f"{STEM}_audit_metrics.json",
        PUBLIC_OUT / f"{STEM}_declaration_gate_metrics.json",
    ]
    refreshed_bundle = refresh_handoff_bundle(bundle_dir, bundle_zip, refresh_candidates)

    handoff_payload = payload(
        "8.7.56.1446",
        "Trial-2 numeric alpha vector Q-ball form-factor unified closure future reopen-trigger closeout expert handoff sync",
        common_inputs,
        [
            row(
                "future_reopen_trigger_closeout_handoff_sync_complete",
                "pass" if closeout_sync_ready else "reject",
                "future reopen-trigger closeout handoff sync complete",
                1 if closeout_sync_ready else 0,
                "Handoff sync is complete only when the closeout sync is ready and the canonical bundle has been refreshed with the closeout artifacts.",
            ),
            row(
                "share_pack_bundle_retained",
                "pass" if share_pack_bundle_available else "reject",
                "canonical share-pack bundle retained for closeout handoff",
                1 if share_pack_bundle_available else 0,
                "The closeout handoff reuses the same canonical Case C / reopen-trigger bundle.",
            ),
            row(
                "share_pack_bundle_refreshed",
                "pass" if refreshed_bundle["copied_count"] > 0 else "reject",
                "canonical share-pack bundle refreshed with closeout files",
                refreshed_bundle["copied_count"],
                "The closeout handoff refreshes the canonical bundle in place so expert readers see the active closeout wording and route state.",
            ),
            row(
                "expert_handoff_note_sync_ready",
                "pass" if inventory_ready else "reject",
                "expert handoff notes sync ready",
                1 if inventory_ready else 0,
                "The expert handoff is ready because the registry, closeout wording, and current notes now point to the same future reopen closeout state.",
            ),
            row(
                "future_reopen_trigger_final_summary_route_selected",
                "pass" if closeout_sync_ready else "reject",
                "future reopen-trigger final summary route selected",
                1 if closeout_sync_ready else 0,
                "After the closeout sync is handed off, the next work is a concise final-summary route for the frozen reopen ordering.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "future_reopen_trigger_closeout_handoff_sync_complete": closeout_sync_ready,
            "share_pack_bundle_zip": display_path(bundle_zip),
            "share_pack_bundle_refresh_count": refreshed_bundle["copied_count"],
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_unified_closure_future_reopen_trigger_closeout_handoff_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "refreshed_bundle_copied_files": [display_path(path) for path in refreshed_bundle["copied_files"]],
            "refreshed_bundle_staging_file_count": refreshed_bundle["staging_file_count"],
            "refreshed_bundle_zip_file_count": refreshed_bundle["zip_file_count"],
            "bundle_zip": display_path(bundle_zip),
            "bundle_dir": display_path(bundle_dir),
            "reopen_trigger_ordering": [PRIMARY_TRIGGER, SECONDARY_TRIGGER, RESERVE_TRIGGER],
        },
    )
    write_artifact("expert_handoff_sync", handoff_payload)

    print("[done] 8.7.56.1443-.1446 artifacts generated")
    print(f"[done] bundle_zip={bundle_zip}")
    print(f"[done] next={NEXT_ROUTE_NAME}")


if __name__ == "__main__":
    main()
