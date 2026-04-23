#!/usr/bin/env python3
"""Generate 8.7.56.1135-.1138 Trial-2 future-canon hold-escalation registry artifacts."""

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

INVENTORY_1123 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_multi_delta_hold_contract_source_inventory_metrics.json"
)
AUDIT_1124 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_multi_delta_hold_contract_audit_metrics.json"
)
GATE_1125 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_multi_delta_hold_contract_declaration_gate_metrics.json"
)
ROUTE_1126 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_eighth_refresh_metrics.json"
INVENTORY_1127 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_carry_over_share_pack_registry_source_inventory_metrics.json"
)
AUDIT_1128 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_carry_over_share_pack_registry_audit_metrics.json"
)
GATE_1129 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_carry_over_share_pack_registry_declaration_gate_metrics.json"
)
ROUTE_1130 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_seventy_ninth_refresh_metrics.json"
INVENTORY_1131 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_hold_handoff_registry_source_inventory_metrics.json"
)
AUDIT_1132 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_hold_handoff_registry_audit_metrics.json"
)
GATE_1133 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_hold_handoff_registry_declaration_gate_metrics.json"
)
ROUTE_1134 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_eightieth_refresh_metrics.json"

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_hold_escalation_registry"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "alpha_is_prediction_future_canon_hold_carry_over_contract"
)
NEXT_ROUTE = "8.7.56.1139"


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


# Function: classify the hold-escalation registry outcome.

def classify(
    registry_ready: bool,
    strong_side_retained: bool,
    hold_policy_frozen: bool,
    future_candidate_retained: bool,
) -> str:
    """Classify the hold-escalation registry outcome."""
    if registry_ready and strong_side_retained and hold_policy_frozen and future_candidate_retained:
        return "future_canon_hold_escalation_registry_frozen"

    if strong_side_retained and hold_policy_frozen and future_candidate_retained:
        return "future_canon_hold_escalation_registry_partial"

    return "future_canon_hold_escalation_registry_incomplete"


# Function: execute the hold-escalation registry branch.

def main() -> None:
    """Execute the Trial-2 future-canon hold-escalation registry branch."""
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
        INVENTORY_1123,
        AUDIT_1124,
        GATE_1125,
        ROUTE_1126,
        INVENTORY_1127,
        AUDIT_1128,
        GATE_1129,
        ROUTE_1130,
        INVENTORY_1131,
        AUDIT_1132,
        GATE_1133,
        ROUTE_1134,
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

    ai_context = read_json(AI_CONTEXT)
    inventory_1123 = read_json(INVENTORY_1123)["summary"]
    audit_1124 = read_json(AUDIT_1124)["summary"]
    gate_1125 = read_json(GATE_1125)["summary"]
    route_1126 = read_json(ROUTE_1126)["summary"]
    inventory_1127 = read_json(INVENTORY_1127)["summary"]
    audit_1128 = read_json(AUDIT_1128)["summary"]
    gate_1129 = read_json(GATE_1129)["summary"]
    route_1130 = read_json(ROUTE_1130)["summary"]
    inventory_1131 = read_json(INVENTORY_1131)["summary"]
    audit_1132 = read_json(AUDIT_1132)["summary"]
    gate_1133 = read_json(GATE_1133)["summary"]
    route_1134 = read_json(ROUTE_1134)["summary"]

    bundle_zip = resolve_path(ai_context["latest_expert_bundle"])
    bundle_dir = resolve_path(ai_context["latest_expert_bundle_dir"])
    bundle_readme = bundle_dir / "README.txt"
    bundle_note = bundle_dir / "BUNDLE_NOTE.txt"
    bundle_questions = bundle_dir / "QUESTIONS_FOR_REVIEW.txt"
    bundle_manifest = bundle_dir / "BUNDLE_MANIFEST.txt"
    for path in (bundle_zip, bundle_dir, bundle_readme, bundle_note, bundle_questions, bundle_manifest):
        require(path)

    bundle_readme_text = read_text(bundle_readme)
    bundle_note_text = read_text(bundle_note)
    bundle_questions_text = read_text(bundle_questions)
    bundle_manifest_text = read_text(bundle_manifest)

    targets = [
        target(status_text, STATUS, "status_1135", "8.7.56.1135", "STATUS must already expose the hold-escalation branch."),
        target(roadmap_text, ROADMAP, "roadmap_1135", "`8.7.56.1135-.1138`", "ROADMAP must already expose the hold-escalation branch."),
        target(part3a_text, PART3A, "part3a_hold_escalation", "future-canon hold-escalation registry", "Part III-A must expose the hold-escalation route."),
        target(part5_text, PART5, "part5_hold_escalation", "future-canon hold-escalation registry", "Part V must expose the hold-escalation route."),
        target(bundle_readme_text, bundle_readme, "bundle_readme_branch", "Next official branch: 8.7.56.1131-.1134 future-canon hold handoff registry.", "The retained share-pack README must expose the immediately previous branch."),
        target(bundle_note_text, bundle_note, "bundle_note_handoff", "prepares the hold state for the next handoff registry", "The retained bundle note must expose the prior handoff reading."),
        target(bundle_questions_text, bundle_questions, "bundle_questions_min_surface", "minimal next public surface", "The retained question pack must still expose the next public-surface question."),
        target(bundle_manifest_text, bundle_manifest, "bundle_manifest_count", "COPIED_COUNT=25", "The retained manifest must preserve the canonical copied-count for the share pack."),
        target(work_history_recent_text, WORK_HISTORY_RECENT, "work_history_recent_1131", "`8.7.56.1131-.1134`", "Recent history must retain the immediately previous completed branch."),
        target(alpha_note_text, NOTE_ALPHA, "alpha_note_formula", "\\alpha = \\frac{c^3}{4\\pi v^2 \\hbar}", "The alpha note must remain in the hold-escalation pack."),
        target(dimension_note_text, NOTE_DIMENSION, "dimension_note_tmchi", "### $T_{M_\\chi}$", "The dimension note must still expose the theorem item."),
        target(si_note_text, NOTE_SI, "si_note_jmu", "$J^\\mu$ の正しい読み方", "The SI note must still expose the source-normalization reserve issue."),
    ]

    share_pack_bundle_available = bool(bundle_zip.exists() and bundle_dir.exists())
    hold_contract_retained = bool(
        inventory_1123["future_canon_multi_delta_hold_contract_ready"]
        and audit_1124["future_canon_multi_delta_hold_contract_ready"]
        and gate_1125["trial2_numeric_alpha_future_canon_multi_delta_hold_contract_completed"]
        and route_1126["future_canon_multi_delta_hold_contract_completed"]
    )
    share_pack_registry_retained = bool(
        inventory_1127["carry_over_share_pack_registry_ready"]
        and audit_1128["carry_over_share_pack_registry_ready"]
        and gate_1129["trial2_numeric_alpha_future_canon_carry_over_share_pack_registry_completed"]
        and route_1130["future_canon_carry_over_share_pack_registry_completed"]
        and route_1130["share_pack_bundle_refreshed"]
    )
    hold_handoff_registry_completed = bool(
        inventory_1131["hold_handoff_registry_ready"]
        and audit_1132["hold_handoff_registry_ready"]
        and gate_1133["trial2_numeric_alpha_future_canon_hold_handoff_registry_completed"]
        and route_1134["future_canon_hold_handoff_registry_completed"]
    )
    next_generation_handoff_rule_frozen = bool(
        gate_1133["trial2_numeric_alpha_next_generation_handoff_rule_frozen"]
        and route_1134["next_generation_handoff_rule_frozen"]
    )
    strong_side_route_state_retained = bool(route_1134["strong_side_route_state"] == "v3_hold_reserve")
    hold_policy_frozen = bool(
        gate_1125["trial2_numeric_alpha_hold_policy_frozen"]
        and route_1126["hold_policy_frozen"]
        and gate_1129["trial2_numeric_alpha_hold_policy_frozen"]
        and route_1130["hold_policy_frozen"]
        and gate_1133["trial2_numeric_alpha_hold_policy_frozen"]
        and route_1134["hold_policy_frozen"]
    )
    future_canon_candidate_retained = bool(
        inventory_1123["future_canon_candidate_retained"]
        and audit_1124["future_canon_candidate_retained"]
        and inventory_1127["future_canon_candidate_retained"]
        and audit_1128["future_canon_candidate_retained"]
        and inventory_1131["future_canon_candidate_retained"]
        and audit_1132["future_canon_candidate_retained"]
        and route_1134["future_canon_candidate_retained"]
    )
    current_canon_not_reopened = bool(
        not route_1126["reopen_prerequisite_satisfied_under_current_canon"]
        and not route_1130["reopen_prerequisite_satisfied_under_current_canon"]
        and not route_1134["reopen_prerequisite_satisfied_under_current_canon"]
    )
    physical_reject_not_selected = bool(
        not route_1126["physical_reject_required"]
        and not route_1130["physical_reject_required"]
        and not route_1134["physical_reject_required"]
    )
    retained_hold_state_ready = bool(
        share_pack_bundle_available
        and hold_contract_retained
        and share_pack_registry_retained
        and hold_handoff_registry_completed
        and next_generation_handoff_rule_frozen
        and strong_side_route_state_retained
        and all(item["present"] for item in targets)
    )
    hold_escalation_registry_ready = bool(
        retained_hold_state_ready
        and hold_policy_frozen
        and future_canon_candidate_retained
        and current_canon_not_reopened
        and physical_reject_not_selected
    )
    hold_carry_over_contract_required = bool(hold_escalation_registry_ready)
    registry_class = classify(
        hold_escalation_registry_ready,
        strong_side_route_state_retained,
        hold_policy_frozen,
        future_canon_candidate_retained,
    )

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
        "prior_1123_json": display_path(INVENTORY_1123),
        "prior_1124_json": display_path(AUDIT_1124),
        "prior_1125_json": display_path(GATE_1125),
        "prior_1126_json": display_path(ROUTE_1126),
        "prior_1127_json": display_path(INVENTORY_1127),
        "prior_1128_json": display_path(AUDIT_1128),
        "prior_1129_json": display_path(GATE_1129),
        "prior_1130_json": display_path(ROUTE_1130),
        "prior_1131_json": display_path(INVENTORY_1131),
        "prior_1132_json": display_path(AUDIT_1132),
        "prior_1133_json": display_path(GATE_1133),
        "prior_1134_json": display_path(ROUTE_1134),
        "share_pack_bundle_zip": display_path(bundle_zip),
        "share_pack_bundle_dir": display_path(bundle_dir),
    }

    inventory = payload(
        "8.7.56.1135",
        "Trial-2 numeric alpha future-canon hold-escalation registry source inventory",
        inputs,
        [
            row("inventory_complete", "pass" if retained_hold_state_ready else "reject", "hold-escalation registry inventory complete", 1 if retained_hold_state_ready else 0, "The hold-escalation inventory is assembled from the frozen hold-contract metrics, the share-pack registry metrics, the hold-handoff metrics, the retained share-pack bundle, the canonical docs, and the retained note pack."),
            row("share_pack_bundle_available", "pass" if share_pack_bundle_available else "reject", "share-pack bundle available for escalation registry", 1 if share_pack_bundle_available else 0, "The retained share-pack bundle must still exist because the hold-escalation registry formalizes that carried source pack."),
            row("next_generation_handoff_rule_frozen", "pass" if next_generation_handoff_rule_frozen else "reject", "next-generation handoff rule remains frozen", 1 if next_generation_handoff_rule_frozen else 0, "The hold-escalation registry starts from the already frozen handoff rule rather than from a reopened computation route."),
            row("strong_side_route_state_retained", "pass" if strong_side_route_state_retained else "reject", "strong-side route state retained", 1 if strong_side_route_state_retained else 0, "The carried strong-side state must remain v3_hold_reserve inside the hold-escalation registry."),
            row("hold_policy_frozen", "pass" if hold_policy_frozen else "reject", "hold-only policy remains frozen", 1 if hold_policy_frozen else 0, "The escalation registry must keep the top-level hold-only interpretation intact."),
            row("future_canon_candidate_retained", "pass" if future_canon_candidate_retained else "reject", "future-canon candidate retained", 1 if future_canon_candidate_retained else 0, "The escalation registry preserves the future-canon candidate rather than collapsing into reject."),
        ],
        {
            "inventory_ready": retained_hold_state_ready,
            "hold_escalation_registry_ready": hold_escalation_registry_ready,
            "retained_hold_state_ready": retained_hold_state_ready,
            "share_pack_bundle_available": share_pack_bundle_available,
            "next_generation_handoff_rule_frozen": next_generation_handoff_rule_frozen,
            "strong_side_route_state_retained": strong_side_route_state_retained,
            "strong_side_route_state": "v3_hold_reserve",
            "hold_policy_frozen": hold_policy_frozen,
            "future_canon_candidate_retained": future_canon_candidate_retained,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "first_route_to_close_or_none": CURRENT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_hold_escalation_inventory_frozen",
            "advance_to_8_7_56_1136": retained_hold_state_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "targets": targets,
            "retained_1123_summary": inventory_1123,
            "retained_1124_summary": audit_1124,
            "retained_1125_summary": gate_1125,
            "retained_1126_summary": route_1126,
            "retained_1127_summary": inventory_1127,
            "retained_1128_summary": audit_1128,
            "retained_1129_summary": gate_1129,
            "retained_1130_summary": route_1130,
            "retained_1131_summary": inventory_1131,
            "retained_1132_summary": audit_1132,
            "retained_1133_summary": gate_1133,
            "retained_1134_summary": route_1134,
        },
    )

    audit = payload(
        "8.7.56.1136",
        "Trial-2 numeric alpha future-canon hold-escalation registry audit",
        inputs,
        [
            row("hold_escalation_registry_ready", "pass" if hold_escalation_registry_ready else "reject", "future-canon hold-escalation registry ready", 1 if hold_escalation_registry_ready else 0, "The hold-escalation registry passes only if the retained hold state remains coherent after the handoff rule is frozen."),
            row("retained_hold_state_honest", "pass" if hold_escalation_registry_ready else "reject", "retained hold state honest", 1 if hold_escalation_registry_ready else 0, "The carried state must remain hold-only, without reopening current canon or triggering physical reject."),
            row("current_canon_not_reopened", "pass" if current_canon_not_reopened else "reject", "current canon not reopened by hold-escalation registry", 1 if current_canon_not_reopened else 0, "The hold-escalation registry formalizes the carried future-canon state and does not restart the current-canon numeric route."),
            row("physical_reject_not_selected", "pass" if physical_reject_not_selected else "reject", "physical reject not selected by hold-escalation registry", 1 if physical_reject_not_selected else 0, "The hold-escalation registry keeps the future-canon candidate live."),
            row("hold_carry_over_contract_required", "pass" if hold_carry_over_contract_required else "reject", "future-canon hold carry-over contract required after escalation registry", 1 if hold_carry_over_contract_required else 0, "Once the hold-escalation registry is frozen, the next honest work is to formalize the downstream hold carry-over contract."),
        ],
        {
            "audit_ready": retained_hold_state_ready,
            "hold_escalation_registry_ready": hold_escalation_registry_ready,
            "retained_hold_state_honest": hold_escalation_registry_ready,
            "share_pack_bundle_available": share_pack_bundle_available,
            "strong_side_route_state_retained": strong_side_route_state_retained,
            "hold_policy_frozen": hold_policy_frozen,
            "future_canon_candidate_retained": future_canon_candidate_retained,
            "reopen_prerequisite_satisfied_under_current_canon": False,
            "physical_reject_required": False,
            "selected_hold_escalation_registry_class": registry_class,
            "first_route_to_close_after_audit_or_none": NEXT_ROUTE_NAME,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_hold_escalation_audited",
            "advance_to_8_7_56_1137": hold_escalation_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"inventory_summary": inventory["summary"]},
    )

    gate = payload(
        "8.7.56.1137",
        "Trial-2 numeric alpha future-canon hold-escalation registry declaration gate",
        inputs,
        [
            row("gate_complete", "pass" if hold_escalation_registry_ready else "reject", "future-canon hold-escalation registry gate complete", 1 if hold_escalation_registry_ready else 0, "The hold-escalation registry becomes official only after the retained hold state and its non-reopen, non-reject reading both pass."),
            row("hold_escalation_registry_completed", "pass" if hold_escalation_registry_ready else "reject", "future-canon hold-escalation registry completed", 1 if hold_escalation_registry_ready else 0, "The declaration gate makes the carried hold / reserve state explicit as one escalation registry."),
            row("hold_policy_retained", "pass" if hold_policy_frozen else "reject", "hold-only policy retained at declaration gate", 1 if hold_policy_frozen else 0, "The declaration gate keeps the top-level hold-only reading intact."),
            row("strong_side_route_state_retained", "pass" if strong_side_route_state_retained else "reject", "strong-side route state retained at declaration gate", 1 if strong_side_route_state_retained else 0, "The declaration gate keeps the v3_hold_reserve state carried rather than escalating it into a reopen."),
            row("next_route_selected", "pass" if hold_carry_over_contract_required else "reject", "future-canon hold carry-over contract selected", 1 if hold_carry_over_contract_required else 0, "The next branch formalizes the downstream hold carry-over contract after the escalation registry is frozen."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "alpha_prediction_future_canon_hold_escalation_registry",
            "trial2_numeric_alpha_future_canon_hold_escalation_registry_completed": hold_escalation_registry_ready,
            "trial2_numeric_alpha_retained_hold_state_ready": retained_hold_state_ready,
            "trial2_numeric_alpha_next_generation_handoff_rule_frozen": next_generation_handoff_rule_frozen,
            "trial2_numeric_alpha_strong_side_route_state_retained": strong_side_route_state_retained,
            "trial2_numeric_alpha_hold_policy_frozen": hold_policy_frozen,
            "trial2_numeric_alpha_future_canon_candidate_retained": future_canon_candidate_retained,
            "trial2_numeric_alpha_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial2_numeric_alpha_physical_reject_required": False,
            "selected_residual_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_future_canon_hold_escalation_gate_closed",
            "advance_to_8_7_56_1138": hold_escalation_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"audit_summary": audit["summary"]},
    )

    route = payload(
        "8.7.56.1138",
        "Trial-2 numeric alpha route contract one-hundred-eighty-first refresh",
        inputs,
        [
            row("route_contract_complete", "pass" if hold_escalation_registry_ready else "reject", "route contract one-hundred-eighty-first refresh complete", 1 if hold_escalation_registry_ready else 0, "The hold-escalation registry is converted into the next downstream hold carry-over contract."),
            row("hold_escalation_registry_completed", "pass" if hold_escalation_registry_ready else "reject", "future-canon hold-escalation registry completed", 1 if hold_escalation_registry_ready else 0, "The retained hold / reserve state is now formalized as the official escalation registry."),
            row("hold_carry_over_contract_selected_as_next_route", "pass" if hold_carry_over_contract_required else "reject", "future-canon hold carry-over contract selected as next route", 1 if hold_carry_over_contract_required else 0, "The next step moves to the downstream hold carry-over contract after the escalation registry is frozen."),
            row("strong_side_reserve_retained", "pass" if strong_side_route_state_retained else "reject", "strong-side reserve retained after hold-escalation registry", 1 if strong_side_route_state_retained else 0, "The reserve-side evidence remains carried in v3 hold reserve state."),
            row("physical_reject_not_selected", "pass" if physical_reject_not_selected else "reject", "physical reject not selected after hold-escalation registry", 1 if physical_reject_not_selected else 0, "The route remains structurally alive after formalizing the escalation registry."),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "future_canon_hold_escalation_registry_completed": hold_escalation_registry_ready,
            "retained_hold_state_ready": retained_hold_state_ready,
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
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_eighty_first_refresh_frozen",
            "advance_to_next_route": hold_escalation_registry_ready,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"gate_summary": gate["summary"], "audit_summary": audit["summary"]},
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_hold_escalation_registry_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_hold_escalation_registry_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "alpha_is_prediction_future_canon_hold_escalation_registry_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_eighty_first_refresh", route)

    print("[done] 8.7.56.1135-.1138 artifacts generated")
    print(f"[bundle] {display_path(bundle_zip)}")
    print(f"[bundle_dir] {display_path(bundle_dir)}")


if __name__ == "__main__":
    main()
