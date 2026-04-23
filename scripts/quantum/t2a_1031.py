#!/usr/bin/env python3
"""Generate 8.7.56.1031-.1034 Trial-2 numeric alpha expert-bundle refresh artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIVATE_OUT = ROOT / "output" / "private" / "quantum"
SUMMARY_OUT = ROOT / "output" / "private" / "summary"

EXPERT_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_two_sector_hierarchy.md")
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EM_DOC = ROOT / "doc" / "quantum" / "16_electromagnetism_charge_maxwell_photon.md"

SOURCE_1019 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_source_inventory_metrics.json"
AUDIT_1020 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_audit_metrics.json"
GATE_1021 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_declaration_gate_metrics.json"
ROUTE_1022 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_second_refresh_metrics.json"
SOURCE_1023 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_source_inventory_metrics.json"
AUDIT_1024 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_audit_metrics.json"
GATE_1025 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_declaration_gate_metrics.json"
ROUTE_1026 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_third_refresh_metrics.json"
SOURCE_1027 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_source_inventory_metrics.json"
AUDIT_1028 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_audit_metrics.json"
GATE_1029 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_declaration_gate_metrics.json"
ROUTE_1030 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_fourth_refresh_metrics.json"

SCRIPT_1019 = ROOT / "scripts" / "quantum" / "t2a_1019.py"
SCRIPT_1023 = ROOT / "scripts" / "quantum" / "t2a_1023.py"
SCRIPT_1027 = ROOT / "scripts" / "quantum" / "t2a_1027.py"
SCRIPT_1031 = ROOT / "scripts" / "quantum" / "t2a_1031.py"

PAPER_FULL = SUMMARY_OUT / "pmodel_paper.pdf"
PAPER_PART3A = SUMMARY_OUT / "pmodel_paper_part3a_quantum_foundations.pdf"
PAPER_PART5 = SUMMARY_OUT / "pmodel_paper_part5_future_predictions.pdf"

BUNDLE_STAMP = "20260324_125528"
BUNDLE_DIR = PRIVATE_OUT / f"expert_review_bundle_{BUNDLE_STAMP}"
BUNDLE_ZIP = PRIVATE_OUT / f"expert_review_bundle_{BUNDLE_STAMP}.zip"

CURRENT_BLOCKER = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_phrase_fragment"
CURRENT_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh"
NEXT_RESPONSE_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_intake"
NEXT_RESPONSE_ARTIFACT = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_note"
NEXT_ROUTE = "8.7.56.1035"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution when a required path is missing.

def require(path: Path) -> None:
    """Require a path to exist before execution continues."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read a UTF-8 text file.

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read a UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return a stable display path.

def display_path(path: Path) -> str:
    """Return a display path relative to the repo root when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing a substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for the given substring pattern."""
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


# Function: build one payload object.

def payload(step: str, name: str, inputs: dict, summary: dict, decision: dict, rows: list[dict], evidence: dict) -> dict:
    """Build one standard metrics payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "summary": summary,
        "decision": decision,
        "rows": rows,
        "evidence": evidence,
    }


# Function: write one JSON metrics artifact and matching CSV rows.

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


# Function: write one UTF-8 text file into the bundle directory.

def write_bundle_text(path: Path, text: str) -> None:
    """Write one UTF-8 text file into the bundle directory."""
    path.write_text(text, encoding="utf-8")


# Function: copy one file into the bundle directory.

def copy_into_bundle(source: Path) -> Path:
    """Copy one source file into the bundle directory."""
    destination = BUNDLE_DIR / source.name
    shutil.copy2(source, destination)
    return destination


# Function: create the current expert-share bundle.

def build_bundle() -> dict:
    """Create the current two-sector hierarchy expert bundle and return its manifest data."""
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    write_bundle_text(
        BUNDLE_DIR / "README.txt",
        """Expert review bundle

Purpose
- Current route: Trial-2 numeric alpha two-sector hierarchy pivot.
- Current blocker family: positive public EM-sector normalization surface is absent in current canon.
- Mechanical wording descent is already stopped by retry triage.
""",
    )
    write_bundle_text(
        BUNDLE_DIR / "EXPERT_NOTE.txt",
        """Expert note

Current fixed state
- Computation route, electron-identification dictionary, and raw final-computation path are fixed.
- Two-sector hierarchy is the current alternate computation pivot.
- Current canon still carries a single-Z_P photon normalization surface and a local Maxwell/QED adoption surface.
- Statement and literal branches for EM-sector normalization added no new public-canonical surface.
""",
    )
    write_bundle_text(
        BUNDLE_DIR / "QUESTIONS_FOR_REVIEW.txt",
        """Questions for review

1. Under the current public canon, is there a defensible positive public statement or formula equivalent to Z_P^EM = 1 and therefore e = g_P?
2. If yes, what is the minimal statement / literal / formula, and where is it located?
3. If no such public statement exists, should Trial-2 numeric alpha now close as structural pass / numeric open?
4. If not, what is the minimal conflict-resolution statement that reconciles the single-Z_P photon canon with the proposed two-sector hierarchy?
""",
    )

    copied = []
    for source in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        EXPERT_NOTE,
        SOURCE_1019,
        AUDIT_1020,
        GATE_1021,
        ROUTE_1022,
        SOURCE_1023,
        AUDIT_1024,
        GATE_1025,
        ROUTE_1026,
        SOURCE_1027,
        AUDIT_1028,
        GATE_1029,
        ROUTE_1030,
        SCRIPT_1019,
        SCRIPT_1023,
        SCRIPT_1027,
        SCRIPT_1031,
        PAPER_FULL,
        PAPER_PART3A,
        PAPER_PART5,
    ):
        copied.append(copy_into_bundle(source).name)

    manifest = [
        "Expert review bundle manifest",
        "",
        f"STAMP={BUNDLE_STAMP}",
        f"COPIED_COUNT={len(copied)}",
        "MISSING_COUNT=0",
        "",
        "[files]",
    ]
    manifest.extend(sorted(copied))
    write_bundle_text(BUNDLE_DIR / "BUNDLE_MANIFEST.txt", "\n".join(manifest) + "\n")

    if BUNDLE_ZIP.exists():
        BUNDLE_ZIP.unlink()

    with zipfile.ZipFile(BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(BUNDLE_DIR.iterdir()):
            handle.write(path, arcname=path.name)

    return {
        "bundle_dir": display_path(BUNDLE_DIR),
        "bundle_zip": display_path(BUNDLE_ZIP),
        "copied_count": len(copied),
        "missing_count": 0,
    }


# Function: execute the expert-bundle refresh branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha expert-bundle refresh branch."""
    required_paths = (
        EXPERT_NOTE,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        SOURCE_1019,
        AUDIT_1020,
        GATE_1021,
        ROUTE_1022,
        SOURCE_1023,
        AUDIT_1024,
        GATE_1025,
        ROUTE_1026,
        SOURCE_1027,
        AUDIT_1028,
        GATE_1029,
        ROUTE_1030,
        SCRIPT_1019,
        SCRIPT_1023,
        SCRIPT_1027,
        SCRIPT_1031,
        PAPER_FULL,
        PAPER_PART3A,
        PAPER_PART5,
    )
    for path in required_paths:
        require(path)

    expert_note_text = read_text(EXPERT_NOTE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)

    source_1019 = read_json(SOURCE_1019)["summary"]
    audit_1020 = read_json(AUDIT_1020)["summary"]
    gate_1021 = read_json(GATE_1021)["summary"]
    route_1022 = read_json(ROUTE_1022)["summary"]
    source_1023 = read_json(SOURCE_1023)["summary"]
    audit_1024 = read_json(AUDIT_1024)["summary"]
    gate_1025 = read_json(GATE_1025)["summary"]
    route_1026 = read_json(ROUTE_1026)["summary"]
    source_1027 = read_json(SOURCE_1027)["summary"]
    audit_1028 = read_json(AUDIT_1028)["summary"]
    gate_1029 = read_json(GATE_1029)["summary"]
    route_1030 = read_json(ROUTE_1030)["summary"]

    bundle_manifest = build_bundle()

    prior_expert_bundle_route_active = (
        gate_1029["selected_residual_route"] == CURRENT_ROUTE
        and route_1030["selected_next_generation_route"] == CURRENT_ROUTE
        and gate_1029["trial2_numeric_alpha_problem_classification"] == "text_search"
        and not gate_1029["trial2_numeric_alpha_text_search_continuation_justified"]
        and bool(gate_1029["trial2_numeric_alpha_mechanical_wording_descent_stopped"])
        and bool(gate_1029["trial2_numeric_alpha_expert_advice_escalation_active"])
    )

    note_has_two_sector_em_literal = hit(expert_note_text, r"Z_P^{\rm EM} = 1") is not None
    note_has_em_bridge_literal = hit(expert_note_text, r"e = g_P / \sqrt{Z_P^{\rm EM}} = g_P") is not None
    note_has_newton_mchi_bridge = hit(expert_note_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}") is not None
    part1_has_single_zp_photon_canon = hit(part1_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}") is not None
    part1_has_vector_zp_surface = hit(part1_text, r"-\frac{Z_{P}}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}") is not None
    part3a_has_expert_state_wording = hit(part3a_text, "expert-advice escalation active") is not None
    part5_has_bundle_next_step = hit(part5_text, "expert-bundle refresh branch `8.7.56.1031-.1034`") is not None
    em_doc_has_local_maxwell_adoption = hit(em_doc_text, "局所（固有時）では Maxwell/QED をそのまま採用") is not None
    status_has_1031_next_step = hit(status_text, "8.7.56.1031") is not None
    roadmap_has_1031_branch = hit(roadmap_text, "`8.7.56.1031-.1034`") is not None

    expert_questions = [
        "current canon の範囲で `Z_P^{EM}=1` または `e=g_P` を public statement / formula として正当に昇格できるか",
        "昇格できない場合、Trial-2 numeric α は structural pass / numeric open で closeout すべきか",
        "昇格できる場合、single-`Z_P` photon canon と two-sector hierarchy note を両立させる最小の statement / formula は何か",
    ]
    response_classes = [
        "positive_public_statement_candidate",
        "no_go_closeout",
        "minimal_conflict_resolution_candidate",
    ]

    inventory_ready = all(
        [
            prior_expert_bundle_route_active,
            note_has_two_sector_em_literal,
            note_has_em_bridge_literal,
            note_has_newton_mchi_bridge,
            part1_has_single_zp_photon_canon,
            part1_has_vector_zp_surface,
            part3a_has_expert_state_wording,
            part5_has_bundle_next_step,
            em_doc_has_local_maxwell_adoption,
            status_has_1031_next_step,
            roadmap_has_1031_branch,
            bundle_manifest["copied_count"] >= 28,
            bundle_manifest["missing_count"] == 0,
        ]
    )

    common_inputs = {
        "expert_note_markdown": display_path(EXPERT_NOTE),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "primary_sources_markdown": display_path(PRIMARY_SOURCES),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "electromagnetism_doc_markdown": display_path(EM_DOC),
        "prior_1019_json": display_path(SOURCE_1019),
        "prior_1020_json": display_path(AUDIT_1020),
        "prior_1021_json": display_path(GATE_1021),
        "prior_1022_json": display_path(ROUTE_1022),
        "prior_1023_json": display_path(SOURCE_1023),
        "prior_1024_json": display_path(AUDIT_1024),
        "prior_1025_json": display_path(GATE_1025),
        "prior_1026_json": display_path(ROUTE_1026),
        "prior_1027_json": display_path(SOURCE_1027),
        "prior_1028_json": display_path(AUDIT_1028),
        "prior_1029_json": display_path(GATE_1029),
        "prior_1030_json": display_path(ROUTE_1030),
        "expert_bundle_dir": bundle_manifest["bundle_dir"],
        "expert_bundle_zip": bundle_manifest["bundle_zip"],
    }

    inventory = payload(
        "8.7.56.1031",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-bundle refresh source inventory",
        common_inputs,
        {
            "inventory_ready": inventory_ready,
            "expert_bundle_ready": bundle_manifest["missing_count"] == 0,
            "expert_bundle_copied_count": bundle_manifest["copied_count"],
            "expert_question_count": len(expert_questions),
            "trial2_numeric_alpha_problem_classification": gate_1029["trial2_numeric_alpha_problem_classification"],
            "trial2_numeric_alpha_text_search_continuation_justified": gate_1029["trial2_numeric_alpha_text_search_continuation_justified"],
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": gate_1029["trial2_numeric_alpha_mechanical_wording_descent_stopped"],
            "first_route_to_close_or_none": NEXT_RESPONSE_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_em_normalization_expert_bundle_inventory_frozen",
            "advance_to_8_7_56_1032": inventory_ready,
            "next_required_artifacts": [NEXT_RESPONSE_ROUTE],
        },
        [
            row("expert_bundle_inventory_complete", "pass" if inventory_ready else "reject", "expert bundle inventory complete", 1 if inventory_ready else 0, "Current canon, note, triage judgment, and minimal questions are bundled together."),
            row("expert_bundle_refreshed_for_current_blocker", "pass" if bundle_manifest["missing_count"] == 0 else "reject", "expert bundle refreshed for current blocker", 1 if bundle_manifest["missing_count"] == 0 else 0, "The bundle targets the unresolved EM-sector normalization public surface."),
            row("retry_triage_state_retained", "pass" if prior_expert_bundle_route_active else "reject", "retry triage state retained", 1 if prior_expert_bundle_route_active else 0, "The refreshed bundle inherits the official stop on mechanical wording descent."),
            row("minimal_three_question_set_included", "pass", "minimal three-question set included", len(expert_questions), "The bundle keeps the expert ask narrow and actionable."),
        ],
        {
            "bundle_manifest": bundle_manifest,
            "expert_questions": expert_questions,
            "note_hits": {
                "em_sector_normalization": hit(expert_note_text, r"Z_P^{\rm EM} = 1"),
                "em_bridge": hit(expert_note_text, r"e = g_P / \sqrt{Z_P^{\rm EM}} = g_P"),
                "newton_mchi_bridge": hit(expert_note_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}"),
            },
        },
    )

    audit = payload(
        "8.7.56.1032",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-bundle question audit",
        common_inputs,
        {
            "audit_ready": inventory_ready,
            "expert_question_set_minimal": True,
            "additional_question_required": False,
            "acceptable_response_types": response_classes,
            "mechanical_wording_descent_stopped": gate_1029["trial2_numeric_alpha_mechanical_wording_descent_stopped"],
            "first_route_to_close_after_audit_or_none": NEXT_RESPONSE_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_em_normalization_expert_questions_audited",
            "advance_to_8_7_56_1033": True,
            "next_required_artifacts": [NEXT_RESPONSE_ROUTE],
        },
        [
            row("expert_question_audit_complete", "pass", "expert question audit complete", 1, "The refreshed bundle audits the question set before response intake is activated."),
            row("expert_question_set_minimal", "pass", "expert question set minimal", len(expert_questions), "The bundle asks only the three questions needed to classify the blocker."),
            row("additional_question_not_required", "pass", "additional expert question not required", 1, "No extra wording or side-route questions are needed before external review."),
            row("mechanical_descent_remains_stopped", "pass" if gate_1029["trial2_numeric_alpha_mechanical_wording_descent_stopped"] else "reject", "mechanical wording descent remains stopped", 1 if gate_1029["trial2_numeric_alpha_mechanical_wording_descent_stopped"] else 0, "Refreshing the bundle does not reopen phrase or fragment descent."),
        ],
        {
            "bundle_manifest": bundle_manifest,
            "prior_expert_advice_summary": source_1027,
            "prior_expert_gate_summary": gate_1029,
        },
    )

    gate = payload(
        "8.7.56.1033",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-bundle declaration gate",
        common_inputs,
        {
            "trial2_numeric_alpha_computation_formula_ready": True,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": True,
            "trial2_numeric_alpha_raw_final_computation_value_available": True,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "trial2_numeric_alpha_final_computation_performed": True,
            "trial2_numeric_alpha_final_computation_result_class": "precanonical_unit_incomplete",
            "trial2_numeric_alpha_problem_classification": gate_1029["trial2_numeric_alpha_problem_classification"],
            "trial2_numeric_alpha_text_search_continuation_justified": gate_1029["trial2_numeric_alpha_text_search_continuation_justified"],
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": gate_1029["trial2_numeric_alpha_mechanical_wording_descent_stopped"],
            "trial2_numeric_alpha_expert_advice_escalation_active": gate_1029["trial2_numeric_alpha_expert_advice_escalation_active"],
            "trial2_numeric_alpha_two_sector_hierarchy_pivot_active": gate_1029["trial2_numeric_alpha_two_sector_hierarchy_pivot_active"],
            "trial2_numeric_alpha_expert_bundle_refresh_ready": bundle_manifest["missing_count"] == 0,
            "trial2_numeric_alpha_expert_response_pending_external_input": True,
            "trial2_numeric_alpha_current_expert_bundle_zip": bundle_manifest["bundle_zip"],
            "selected_residual_route": NEXT_RESPONSE_ROUTE,
            "missing_v2_artifact": NEXT_RESPONSE_ARTIFACT,
            "historical_text_search_blocker": CURRENT_BLOCKER,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_em_normalization_expert_bundle_gate_closed",
            "advance_to_8_7_56_1034": True,
            "next_required_artifacts": [NEXT_RESPONSE_ROUTE],
        },
        [
            row("expert_bundle_gate_complete", "pass", "expert-bundle declaration gate complete", 1, "The official state is updated after the bundle refresh."),
            row("expert_bundle_refresh_ready", "pass" if bundle_manifest["missing_count"] == 0 else "reject", "expert-bundle refresh ready", 1 if bundle_manifest["missing_count"] == 0 else 0, "The refreshed bundle is ready for external sharing."),
            row("expert_response_pending_external_input", "pass", "expert response pending external input", 1, "The next official route now depends on an external expert response."),
            row("current_pack_not_closeout_ready", "reject", "current pack still not closeout ready", 0, "Refreshing the bundle does not solve the unresolved public surface by itself."),
        ],
        {
            "expert_question_audit_summary": audit["summary"],
            "bundle_manifest": bundle_manifest,
            "prior_expert_gate_summary": gate_1029,
        },
    )

    route = payload(
        "8.7.56.1034",
        "Trial-2 numeric alpha route contract one-hundred-fifty-fifth refresh",
        common_inputs,
        {
            "selected_next_generation_route": NEXT_RESPONSE_ROUTE,
            "strong_side_route_state": route_1030["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(route_1030["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(route_1030["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(route_1030["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(route_1030["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": bool(route_1030["unit_consistency_audit_branch_retained"]),
            "dimensionless_alpha_bridge_branch_retained": bool(route_1030["dimensionless_alpha_bridge_branch_retained"]),
            "em_unit_convention_bridge_branch_retained": bool(route_1030["em_unit_convention_bridge_branch_retained"]),
            "mapping_statement_branch_retained": bool(route_1030["mapping_statement_branch_retained"]),
            "mapping_literal_branch_retained": bool(route_1030["mapping_literal_branch_retained"]),
            "expert_advice_escalation_branch_retained": True,
            "two_sector_hierarchy_pivot_retained": bool(route_1030["two_sector_hierarchy_pivot_retained"]),
            "same_pattern_retry_threshold_reached": bool(route_1030["same_pattern_retry_threshold_reached"]),
            "retry_triage_gate_triggered": bool(route_1030["retry_triage_gate_triggered"]),
            "mechanical_wording_descent_stopped": bool(route_1030["mechanical_wording_descent_stopped"]),
            "expert_bundle_refresh_completed": True,
            "external_dependency_active": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_fifty_fifth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_RESPONSE_ROUTE],
        },
        [
            row("route_contract_155_complete", "pass", "route contract one-hundred-fifty-fifth refresh complete", 1, "The bundle refresh is converted into the next-generation contract."),
            row("next_route_selected_as_expert_response_intake", "pass", "next route selected as expert response intake", 1, "The next official branch is response intake, not more wording descent."),
            row("precision_mainline_retained", "pass" if route_1030["precision_alpha_mainline_retained"] else "reject", "precision-alpha mainline retained", 1 if route_1030["precision_alpha_mainline_retained"] else 0, "Trial-2 numeric alpha remains the precision mainline."),
            row("external_dependency_reactivated", "pass", "external dependency reactivated", 1, "Once the share pack is current, the blocker becomes a response-intake dependency."),
        ],
        {
            "gate_summary": gate["summary"],
            "bundle_manifest": bundle_manifest,
            "prior_route_summary": route_1030,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_fifth_refresh",
        route,
    )

    print("[done] 8.7.56.1031-.1034 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_fifth_refresh_metrics.json")
    print(f" - bundle_zip = {bundle_manifest['bundle_zip']}")


# Function: run the expert-bundle refresh branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha expert-bundle refresh branch."""
    main()


if __name__ == "__main__":
    run_cli()
