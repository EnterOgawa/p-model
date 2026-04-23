#!/usr/bin/env python3
"""Generate 8.7.56.1147-.1150 Trial-2 future-canon reserve hold rule artifacts."""

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
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
NOTE_ALPHA = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_alpha_is_prediction.md")
NOTE_DIMENSION = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_dimension_normalization_review.md")
NOTE_SI = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_si_dimension_tracking.md")

ROUTE_1138 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_eighty_first_refresh_metrics.json"
ROUTE_1142 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_eighty_second_refresh_metrics.json"
INVENTORY_1143 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_reserve_carry_over_contract_source_inventory_metrics.json"
)
AUDIT_1144 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_reserve_carry_over_contract_audit_metrics.json"
)
GATE_1145 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_reserve_carry_over_contract_declaration_gate_metrics.json"
)
ROUTE_1146 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_eighty_third_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_reserve_hold_rule"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_reserve_contract"
)
NEXT_ROUTE = "8.7.56.1151"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort when one required path is missing.

def require(path: Path) -> None:
    """Abort when one required file or directory is missing."""
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


# Function: resolve one repo-relative or absolute path string.

def resolve_path(path_str: str) -> Path:
    """Resolve one repo-relative or absolute path string."""
    path = Path(path_str)
    if path.is_absolute():
        return path

    return ROOT / path


# Function: return one stable display path.

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing one substring.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# Function: build one standard metrics payload.

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
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


# Function: write one JSON metrics artifact and one CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one metrics payload as JSON and CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build one wording target record.

def target(text: str, path: Path, key: str, pattern: str, note: str) -> dict:
    """Build one wording target record."""
    evidence = hit(text, pattern)
    return {
        "file_key": key,
        "file": display_path(path),
        "pattern": pattern,
        "present": evidence is not None,
        "note": note,
        "evidence": evidence,
    }


# Function: classify the reserve hold rule outcome.

def classify(rule_ready: bool, hold_policy_frozen: bool, future_candidate_retained: bool) -> str:
    """Classify the reserve hold rule outcome."""
    if rule_ready and hold_policy_frozen and future_candidate_retained:
        return "future_canon_reserve_hold_rule_frozen"

    if hold_policy_frozen and future_candidate_retained:
        return "future_canon_reserve_hold_rule_partial"

    return "future_canon_reserve_hold_rule_incomplete"


# Function: execute the reserve hold rule branch.

def main() -> None:
    """Execute the Trial-2 future-canon reserve hold rule branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART3A,
        PART5,
        NOTE_ALPHA,
        NOTE_DIMENSION,
        NOTE_SI,
        ROUTE_1138,
        ROUTE_1142,
        INVENTORY_1143,
        AUDIT_1144,
        GATE_1145,
        ROUTE_1146,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    alpha_note_text = read_text(NOTE_ALPHA)
    dimension_note_text = read_text(NOTE_DIMENSION)
    si_note_text = read_text(NOTE_SI)

    route_1138 = read_json(ROUTE_1138)["summary"]
    route_1142 = read_json(ROUTE_1142)["summary"]
    inventory_1143 = read_json(INVENTORY_1143)["summary"]
    audit_1144 = read_json(AUDIT_1144)["summary"]
    gate_1145 = read_json(GATE_1145)["summary"]
    route_1146 = read_json(ROUTE_1146)["summary"]
    ai_context = read_json(AI_CONTEXT)

    bundle_zip = resolve_path(ai_context["latest_expert_bundle"])
    bundle_dir = resolve_path(ai_context["latest_expert_bundle_dir"])
    bundle_readme = bundle_dir / "README.txt"
    bundle_note = bundle_dir / "BUNDLE_NOTE.txt"
    bundle_questions = bundle_dir / "QUESTIONS_FOR_REVIEW.txt"
    bundle_manifest = bundle_dir / "BUNDLE_MANIFEST.txt"
    for path in (bundle_zip, bundle_dir, bundle_readme, bundle_note, bundle_questions, bundle_manifest):
        require(path)

    targets = [
        target(status_text, STATUS, "status_1147", "8.7.56.1147", "STATUS must already expose the reserve hold-rule branch."),
        target(roadmap_text, ROADMAP, "roadmap_1147", "`8.7.56.1147-.1150`", "ROADMAP must already expose the reserve hold-rule branch."),
        target(part3a_text, PART3A, "part3a_reserve_hold_rule", "future-canon reserve hold rule", "Part III-A must expose the reserve hold-rule route."),
        target(part5_text, PART5, "part5_reserve_hold_rule", "future-canon reserve hold rule", "Part V must expose the reserve hold-rule route."),
        target(read_text(bundle_readme), bundle_readme, "bundle_readme_branch", "Next official branch: 8.7.56.1131-.1134 future-canon hold handoff registry.", "The retained share-pack README must preserve bundle provenance."),
        target(read_text(bundle_note), bundle_note, "bundle_note_handoff", "prepares the hold state for the next handoff registry", "The retained bundle note must preserve the handoff reading."),
        target(read_text(bundle_questions), bundle_questions, "bundle_questions_min_surface", "minimal next public surface", "The retained question pack must still expose the next public-surface question."),
        target(read_text(bundle_manifest), bundle_manifest, "bundle_manifest_count", "COPIED_COUNT=25", "The retained manifest must preserve the copied-count."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "work_history_recent_1143", "`8.7.56.1143-.1146`", "Recent history must retain the immediately previous branch."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_formula", "\\alpha = \\frac{c^3}{4\\pi v^2 \\hbar}", "The alpha note must remain in the reserve hold-rule pack."),
        target(dimension_note_text, NOTE_DIMENSION, "dimension_note_case_c", "### Case C: $T_{M_\\chi}$ no-go", "The dimension note must still expose the no-go reserve surface."),
        target(si_note_text, NOTE_SI, "si_note_jmu", "$J^\\mu$ の正しい読み方", "The SI note must still expose the reserve-side issue."),
    ]

    share_pack_bundle_available = bool(bundle_zip.exists() and bundle_dir.exists())
    hold_escalation_registry_retained = bool(route_1138["future_canon_hold_escalation_registry_completed"])
    hold_carry_over_contract_completed = bool(route_1142["future_canon_hold_carry_over_contract_completed"])
    reserve_carry_over_contract_completed = bool(
        inventory_1143["reserve_carry_over_contract_ready"]
        and audit_1144["reserve_carry_over_contract_ready"]
        and gate_1145["trial2_numeric_alpha_future_canon_reserve_carry_over_contract_completed"]
        and route_1146["future_canon_reserve_carry_over_contract_completed"]
    )
    next_generation_handoff_rule_frozen = bool(
        route_1138["next_generation_handoff_rule_frozen"]
        and route_1142["next_generation_handoff_rule_frozen"]
        and inventory_1143["next_generation_handoff_rule_frozen"]
        and route_1146["next_generation_handoff_rule_frozen"]
    )
    strong_side_route_state_retained = bool(
        route_1138["strong_side_route_state"] == "v3_hold_reserve"
        and route_1142["strong_side_route_state"] == "v3_hold_reserve"
        and inventory_1143["strong_side_route_state"] == "v3_hold_reserve"
        and route_1146["strong_side_route_state"] == "v3_hold_reserve"
    )
    hold_policy_frozen = bool(route_1142["hold_policy_frozen"] and route_1146["hold_policy_frozen"])
    future_canon_candidate_retained = bool(route_1146["future_canon_candidate_retained"])
    current_canon_not_reopened = bool(
        not route_1142["reopen_prerequisite_satisfied_under_current_canon"]
        and not route_1146["reopen_prerequisite_satisfied_under_current_canon"]
    )
    physical_reject_not_selected = bool(
        not route_1142["physical_reject_required"] and not route_1146["physical_reject_required"]
    )
    reserve_side_rule_ready = bool(
        share_pack_bundle_available
        and hold_escalation_registry_retained
        and hold_carry_over_contract_completed
        and reserve_carry_over_contract_completed
        and next_generation_handoff_rule_frozen
        and strong_side_route_state_retained
        and all(item["present"] for item in targets)
    )
    reserve_hold_rule_ready = bool(
        reserve_side_rule_ready
        and hold_policy_frozen
        and future_canon_candidate_retained
        and current_canon_not_reopened
        and physical_reject_not_selected
    )
    reserve_contract_required = bool(reserve_hold_rule_ready)
    rule_class = classify(reserve_hold_rule_ready, hold_policy_frozen, future_canon_candidate_retained)

    inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "work_history_recent_markdown": display_path(WORK_HISTORY_RECENT),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "alpha_prediction_note": display_path(NOTE_ALPHA),
        "dimension_normalization_review_note": display_path(NOTE_DIMENSION),
        "si_dimension_tracking_note": display_path(NOTE_SI),
        "prior_1138_json": display_path(ROUTE_1138),
        "prior_1142_json": display_path(ROUTE_1142),
        "prior_1143_json": display_path(INVENTORY_1143),
        "prior_1144_json": display_path(AUDIT_1144),
        "prior_1145_json": display_path(GATE_1145),
        "prior_1146_json": display_path(ROUTE_1146),
        "share_pack_bundle_zip": display_path(bundle_zip),
        "share_pack_bundle_dir": display_path(bundle_dir),
    }

    inventory = payload(
        "8.7.56.1147",
        "Trial-2 numeric alpha future-canon reserve hold rule source inventory",
        inputs,
        [
            row("inventory_complete", "pass" if reserve_side_rule_ready else "reject", "reserve hold-rule inventory complete", 1 if reserve_side_rule_ready else 0, "The reserve hold-rule inventory is assembled from the frozen reserve carry-over metrics, the retained hold metrics, the retained share-pack bundle, the canonical docs, and the retained note pack."),
            row("share_pack_bundle_available", "pass" if share_pack_bundle_available else "reject", "share-pack bundle available for reserve hold rule", 1 if share_pack_bundle_available else 0, "The retained share-pack bundle must still exist because the reserve hold rule inherits that carried source pack."),
            row("reserve_carry_over_contract_completed", "pass" if reserve_carry_over_contract_completed else "reject", "reserve carry-over contract remains completed", 1 if reserve_carry_over_contract_completed else 0, "The reserve hold rule starts from the already completed downstream reserve carry-over contract."),
            row("strong_side_route_state_retained", "pass" if strong_side_route_state_retained else "reject", "strong-side route state retained", 1 if strong_side_route_state_retained else 0, "The carried strong-side state must remain v3_hold_reserve inside the reserve hold rule."),
            row("hold_policy_frozen", "pass" if hold_policy_frozen else "reject", "hold-only policy remains frozen", 1 if hold_policy_frozen else 0, "The reserve hold rule must keep the top-level hold-only interpretation intact."),
        ],
        {
            "inventory_ready": reserve_side_rule_ready,
            "reserve_hold_rule_ready": reserve_hold_rule_ready,
            "reserve_side_rule_ready": reserve_side_rule_ready,
            "share_pack_bundle_available": share_pack_bundle_available,
            "hold_escalation_registry_completed": hold_escalation_registry_retained,
            "hold_carry_over_contract_completed": hold_carry_over_contract_completed,
            "reserve_carry_over_contract_completed": reserve_carry_over_contract_completed,
            "next_generation_handoff_rule_frozen": next_generation_handoff_rule_frozen,
            "strong_side_route_state_retained": strong_side_route_state_retained,
            "strong_side_route_state": "v3_hold_reserve",
            "hold_policy_frozen": hold_policy_frozen,
            "future_canon_candidate_retained": future_canon_candidate_retained,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {"overall_status": "trial2_numeric_alpha_future_canon_reserve_hold_rule_inventory_frozen", "advance_to_8_7_56_1148": reserve_side_rule_ready, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"targets": targets, "retained_1143_summary": inventory_1143, "retained_1144_summary": audit_1144, "retained_1145_summary": gate_1145, "retained_1146_summary": route_1146},
    )

    audit = payload(
        "8.7.56.1148",
        "Trial-2 numeric alpha future-canon reserve hold rule audit",
        inputs,
        [
            row("reserve_hold_rule_ready", "pass" if reserve_hold_rule_ready else "reject", "future-canon reserve hold rule ready", 1 if reserve_hold_rule_ready else 0, "The reserve hold rule passes only if the reserve-side rule remains coherent after the reserve carry-over contract is frozen."),
            row("reserve_hold_rule_honest", "pass" if reserve_hold_rule_ready else "reject", "reserve-side hold rule honest", 1 if reserve_hold_rule_ready else 0, "The reserve-side hold rule must remain hold-only, without reopening current canon or triggering physical reject."),
            row("current_canon_not_reopened", "pass" if current_canon_not_reopened else "reject", "current canon not reopened by reserve hold rule", 1 if current_canon_not_reopened else 0, "The reserve hold rule formalizes the carried future-canon state and does not restart the current-canon numeric route."),
            row("physical_reject_not_selected", "pass" if physical_reject_not_selected else "reject", "physical reject not selected by reserve hold rule", 1 if physical_reject_not_selected else 0, "The reserve hold rule keeps the future-canon candidate live."),
            row("reserve_contract_required", "pass" if reserve_contract_required else "reject", "future-canon reserve contract required after reserve hold rule", 1 if reserve_contract_required else 0, "Once the reserve hold rule is frozen, the next honest work is the downstream reserve contract."),
        ],
        {
            "audit_ready": reserve_side_rule_ready,
            "reserve_hold_rule_ready": reserve_hold_rule_ready,
            "reserve_hold_rule_honest": reserve_hold_rule_ready,
            "share_pack_bundle_available": share_pack_bundle_available,
            "strong_side_route_state_retained": strong_side_route_state_retained,
            "hold_policy_frozen": hold_policy_frozen,
            "future_canon_candidate_retained": future_canon_candidate_retained,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "selected_reserve_hold_rule_class": rule_class,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {"overall_status": "trial2_numeric_alpha_future_canon_reserve_hold_rule_audited", "advance_to_8_7_56_1149": reserve_hold_rule_ready, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"inventory_summary": inventory["summary"]},
    )

    gate = payload(
        "8.7.56.1149",
        "Trial-2 numeric alpha future-canon reserve hold rule declaration gate",
        inputs,
        [
            row("gate_complete", "pass" if reserve_hold_rule_ready else "reject", "future-canon reserve hold rule gate complete", 1 if reserve_hold_rule_ready else 0, "The reserve hold rule becomes official only after the reserve-side rule and its non-reopen, non-reject reading both pass."),
            row("reserve_hold_rule_completed", "pass" if reserve_hold_rule_ready else "reject", "future-canon reserve hold rule completed", 1 if reserve_hold_rule_ready else 0, "The declaration gate makes the reserve-side hold rule explicit as one downstream reserve rule."),
            row("hold_policy_retained", "pass" if hold_policy_frozen else "reject", "hold-only policy retained at declaration gate", 1 if hold_policy_frozen else 0, "The declaration gate keeps the top-level hold-only reading intact."),
            row("strong_side_route_state_retained", "pass" if strong_side_route_state_retained else "reject", "strong-side route state retained at declaration gate", 1 if strong_side_route_state_retained else 0, "The declaration gate keeps the v3_hold_reserve state carried rather than escalating it into a reopen."),
            row("next_route_selected", "pass" if reserve_contract_required else "reject", "future-canon reserve contract selected", 1 if reserve_contract_required else 0, "The next branch formalizes the downstream reserve contract after the reserve hold rule is frozen."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_future_canon_reserve_hold_rule",
            "trial2_numeric_alpha_future_canon_reserve_hold_rule_completed": reserve_hold_rule_ready,
            "trial2_numeric_alpha_reserve_hold_rule_ready": reserve_hold_rule_ready,
            "trial2_numeric_alpha_reserve_carry_over_contract_completed": reserve_carry_over_contract_completed,
            "trial2_numeric_alpha_next_generation_handoff_rule_frozen": next_generation_handoff_rule_frozen,
            "trial2_numeric_alpha_strong_side_route_state_retained": strong_side_route_state_retained,
            "trial2_numeric_alpha_hold_policy_frozen": hold_policy_frozen,
            "trial2_numeric_alpha_future_canon_candidate_retained": future_canon_candidate_retained,
            "trial2_numeric_alpha_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial2_numeric_alpha_physical_reject_required": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {"overall_status": "trial2_numeric_alpha_future_canon_reserve_hold_rule_gate_closed", "advance_to_8_7_56_1150": reserve_hold_rule_ready, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"audit_summary": audit["summary"]},
    )

    route = payload(
        "8.7.56.1150",
        "Trial-2 numeric alpha route contract one-hundred-eighty-fourth refresh",
        inputs,
        [
            row("route_contract_complete", "pass" if reserve_hold_rule_ready else "reject", "route contract one-hundred-eighty-fourth refresh complete", 1 if reserve_hold_rule_ready else 0, "The reserve hold rule is converted into the next downstream reserve contract."),
            row("reserve_hold_rule_completed", "pass" if reserve_hold_rule_ready else "reject", "future-canon reserve hold rule completed", 1 if reserve_hold_rule_ready else 0, "The reserve-side hold rule is now formalized as the official downstream reserve rule."),
            row("reserve_contract_selected_as_next_route", "pass" if reserve_contract_required else "reject", "future-canon reserve contract selected as next route", 1 if reserve_contract_required else 0, "The next step moves to the downstream reserve contract after the reserve hold rule is frozen."),
            row("strong_side_reserve_retained", "pass" if strong_side_route_state_retained else "reject", "strong-side reserve retained after reserve hold rule", 1 if strong_side_route_state_retained else 0, "The reserve-side evidence remains carried in v3 hold reserve state."),
            row("physical_reject_not_selected", "pass" if physical_reject_not_selected else "reject", "physical reject not selected after reserve hold rule", 1 if physical_reject_not_selected else 0, "The route remains structurally alive after formalizing the reserve hold rule."),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "future_canon_reserve_hold_rule_completed": reserve_hold_rule_ready,
            "reserve_hold_rule_ready": reserve_hold_rule_ready,
            "reserve_carry_over_contract_completed": reserve_carry_over_contract_completed,
            "next_generation_handoff_rule_frozen": next_generation_handoff_rule_frozen,
            "strong_side_route_state": "v3_hold_reserve",
            "future_canon_candidate_retained": future_canon_candidate_retained,
            "hold_policy_frozen": hold_policy_frozen,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "source_bundle_zip": display_path(bundle_zip),
            "source_bundle_dir": display_path(bundle_dir),
        },
        {"overall_status": "trial2_numeric_alpha_route_contract_one_hundred_eighty_fourth_refresh_frozen", "advance_to_next_route": reserve_hold_rule_ready, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"gate_summary": gate["summary"], "audit_summary": audit["summary"]},
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_alpha_is_prediction_future_canon_reserve_hold_rule_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_alpha_is_prediction_future_canon_reserve_hold_rule_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_alpha_is_prediction_future_canon_reserve_hold_rule_declaration_gate", gate)
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_eighty_fourth_refresh", route)

    print("[done] 8.7.56.1147-.1150 artifacts generated")
    print(f"[bundle] {display_path(bundle_zip)}")
    print(f"[bundle_dir] {display_path(bundle_dir)}")


if __name__ == "__main__":
    main()
