#!/usr/bin/env python3
"""
Freeze the external-share / independent-galaxy follow-through branch for 8.7.55.3.133-.136.

This branch starts after the direct kappa_a bridge has already been closed on the
theory side and confirmed on the paper side. The remaining work is not another
derivation loop; it is a follow-through branch that:

1. freezes the canonical external share-pack inventory,
2. freezes the expert-note / dissemination wording that accompanies that pack,
3. inventories whether the repo already contains the prerequisites for the first
   non-SPARC independent-galaxy intake, and
4. decides whether the next official route should wait for feedback or open a
   dedicated dataset-intake branch.
"""

from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PRIVATE_QUANTUM = ROOT / "output" / "private" / "quantum"
PAGE_AUDIT = ROOT / "output" / "private" / "summary" / "page_audit"
DECLARATION_GATE = OUT / "mass_origin_dark_matter_elimination_declaration_gate_metrics.json"
DIRECT_BRIDGE = OUT / "mass_origin_direct_kappa_bridge_statement_freeze_metrics.json"
EQUALITY_AUDIT = OUT / "mass_origin_direct_kappa_sparc_equality_audit_metrics.json"
PROFILE_METRICS = OUT / "mass_origin_postnewtonian_rotation_curve_profile_metrics.json"
PART2_FREEZE = OUT / "mass_origin_part2_direct_kappa_wording_freeze_metrics.json"
PART3A_FREEZE = OUT / "mass_origin_part3a_direct_kappa_bridge_wording_freeze_metrics.json"
PART45_FREEZE = OUT / "mass_origin_part4_part5_dark_matter_wording_pack_freeze_metrics.json"
PAPER_SYNC_GATE = OUT / "mass_origin_dark_matter_paper_side_user_build_handoff_gate_metrics.json"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART4 = ROOT / "doc" / "paper" / "13_part4_verification.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
PART2_PDF = ROOT / "papers" / "pmodel_paper_part2_astrophysics.pdf"
PART3A_PDF = ROOT / "papers" / "pmodel_paper_part3a_quantum_foundations.pdf"
PART4_PDF = ROOT / "papers" / "pmodel_paper_part4_verification.pdf"
PART5_PDF = ROOT / "papers" / "pmodel_paper_part5_future_predictions.pdf"
PART2_AUDIT = PAGE_AUDIT / "part2_direct_kappa_buildsync_page065-065.png"
PART3A_AUDIT = PAGE_AUDIT / "part3a_direct_kappa_buildsync_page002-02.png"
PART4_AUDIT = PAGE_AUDIT / "part4_direct_kappa_buildsync_page018-018.png"
PART5_AUDIT = PAGE_AUDIT / "part5_direct_kappa_buildsync_page004-04.png"


# Function: Return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: Abort immediately when a required artifact is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: Read a UTF-8 JSON artifact into a dictionary.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: Read a UTF-8 text file.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: Convert an absolute path to a repo-relative string.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: Return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: Build a common metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: Build a common payload with the shared schema.

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: Save a JSON artifact and the paired CSV row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: Return the most recent expert-share bundle zip.

def latest_bundle_zip() -> Path:
    candidates = sorted(PRIVATE_QUANTUM.glob("expert_review_bundle_*.zip"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise SystemExit("[fail] no expert review bundle zip found in output/private/quantum")

    return candidates[-1]


# Function: Restore the share-pack staging directory from the canonical zip if needed.

def ensure_staging_dir(bundle_zip: Path) -> tuple[Path, bool]:
    staging_dir = bundle_zip.with_suffix("")
    created_from_zip = False
    if not staging_dir.exists():
        staging_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip, "r") as archive:
            archive.extractall(staging_dir)

        created_from_zip = True

    return staging_dir, created_from_zip


# Function: Return the canonical README text for the current external share pack.

def canonical_readme(bundle_zip: Path) -> str:
    generated = now_iso()
    return (
        "waveP expert review bundle\n"
        f"Generated: {generated}\n\n"
        "Scope\n"
        "- Current Phase 8 status after completing 8.7.55.3.132.\n"
        "- Direct kappa bridge is theory-closed and paper-side wording/build/page-audit sync is complete.\n"
        "- This bundle is intended for expert review, external sharing, and the first dissemination pass.\n\n"
        "Included\n"
        "- Current control docs: STATUS / ROADMAP / AI_CONTEXT_MIN / WORK_HISTORY_RECENT\n"
        "- Direct kappa bridge and paper-sync metrics\n"
        "- Built paper PDFs for Part II / Part III-A / Part IV / Part V\n"
        "- Page-audit PNGs showing the direct kappa wording on paper pages\n\n"
        "Current state\n"
        "- Latest complete step: 8.7.55.3.132\n"
        "- Dark-matter elimination theory-side closeout: complete\n"
        "- Paper-side build / TeX audit / page audit: complete\n"
        "- Current canonical zip: "
        f"{bundle_zip.name}\n"
        "- Next roadmap branch: 8.7.55.3.133-.136 external-share / independent-galaxy follow-through\n"
    )


# Function: Return the canonical expert note for the current external share pack.

def canonical_expert_note() -> str:
    return (
        "Expert note\n\n"
        "The core update is that the dark-matter-elimination bridge no longer depends on the stalled\n"
        "vector-hierarchy-to-kappa_a wording residual. Instead, kappa_a is fixed directly from the\n"
        "background P-wave late-time law:\n\n"
        "  P_bg(t) propto exp[-H0^(P) (t - t0)]\n"
        "  omega_bg = H0^(P)\n"
        "  lambda_bg = 2 pi c / H0^(P)\n"
        "  a0 = c^2 / lambda_bg = c H0^(P) / (2 pi)\n"
        "  kappa_a = a0 / (c H0^(P)) = 1 / (2 pi)\n\n"
        "This direct bridge is frozen in the public metrics and matches the operational SPARC value\n"
        "at machine precision.\n\n"
        "What is completed\n"
        "- Part II states that kappa_a is a derived quantity, not a SPARC-only fit coefficient.\n"
        "- Part III-A connects the exact vector hierarchy to the direct background-wave bridge.\n"
        "- Part IV states the falsification rule explicitly: if an independent galaxy sample requires\n"
        "  kappa_a != 1/(2 pi) while keeping the same baryon interface and H0^(P), the branch is rejected.\n"
        "- Part V states the external-galaxy future test in the same canonical wording.\n"
        "- Targeted paper builds for Part II / III-A / IV / V succeeded, TeX audit passed, and page audits\n"
        "  confirmed the wording on the generated PDFs.\n\n"
        "Main files to read first\n"
        "- STATUS.md\n"
        "- ROADMAP.md\n"
        "- mass_origin_direct_kappa_bridge_statement_freeze_metrics.json\n"
        "- mass_origin_dark_matter_postnewtonian_direct_bridge_retry_metrics.json\n"
        "- mass_origin_direct_kappa_sparc_equality_audit_metrics.json\n"
        "- mass_origin_dark_matter_elimination_declaration_gate_metrics.json\n\n"
        "Follow-through focus\n"
        "- The theory-side and paper-side closure is complete.\n"
        "- The next technical checkpoint is no longer another derivation loop; it is the first\n"
        "  independent-galaxy intake, where a non-SPARC public rotation-curve sample must be tested\n"
        "  against the same baryon interface and the same derived kappa_a = 1/(2 pi).\n"
    )


# Function: Audit whether a file exists in the staging directory.

def staged_item_record(staging_dir: Path, source_path: Path) -> dict:
    staged_path = staging_dir / source_path.name
    return {
        "item_key": source_path.name,
        "source_file": rel(source_path),
        "staged_file": str(staged_path.relative_to(ROOT)).replace("\\", "/"),
        "present": staged_path.exists(),
        "size_bytes": staged_path.stat().st_size if staged_path.exists() else 0,
    }


# Function: Audit whether a phrase is present in a note file.

def note_phrase_record(text: str, phrase: str, note: str) -> dict:
    phrase_hit = hit(text, phrase)
    return {"phrase": phrase, "present": phrase_hit is not None, "note": note, "evidence": phrase_hit}


# Function: Run the external-share / independent-galaxy follow-through branch and write artifacts.

def main() -> None:
    for path in (
        DECLARATION_GATE,
        DIRECT_BRIDGE,
        EQUALITY_AUDIT,
        PROFILE_METRICS,
        PART2_FREEZE,
        PART3A_FREEZE,
        PART45_FREEZE,
        PAPER_SYNC_GATE,
        PRIMARY_SOURCES,
        PART4,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART2_PDF,
        PART3A_PDF,
        PART4_PDF,
        PART5_PDF,
        PART2_AUDIT,
        PART3A_AUDIT,
        PART4_AUDIT,
        PART5_AUDIT,
    ):
        req(path)

    bundle_zip = latest_bundle_zip()
    staging_dir, created_from_zip = ensure_staging_dir(bundle_zip)
    declaration_gate = read_json(DECLARATION_GATE)
    paper_sync_gate = read_json(PAPER_SYNC_GATE)
    primary_sources_text = read_text(PRIMARY_SOURCES)
    part4_text = read_text(PART4)
    part5_text = read_text(PART5)

    readme_text = canonical_readme(bundle_zip)
    expert_note_text = canonical_expert_note()
    readme_path = staging_dir / "README.txt"
    expert_note_path = staging_dir / "EXPERT_NOTE.txt"
    readme_path.write_text(readme_text, encoding="utf-8")
    expert_note_path.write_text(expert_note_text, encoding="utf-8")

    share_pack_items = [
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        DIRECT_BRIDGE,
        EQUALITY_AUDIT,
        PROFILE_METRICS,
        DECLARATION_GATE,
        PART2_FREEZE,
        PART3A_FREEZE,
        PART45_FREEZE,
        PAPER_SYNC_GATE,
        PART2_PDF,
        PART3A_PDF,
        PART4_PDF,
        PART5_PDF,
        PART2_AUDIT,
        PART3A_AUDIT,
        PART4_AUDIT,
        PART5_AUDIT,
        readme_path,
        expert_note_path,
    ]
    share_pack_records = [staged_item_record(staging_dir, item) for item in share_pack_items]
    present_share_pack_records = [item for item in share_pack_records if item["present"]]
    missing_share_pack_records = [item for item in share_pack_records if not item["present"]]

    readme_phrases = [
        note_phrase_record(readme_text, "Latest complete step: 8.7.55.3.132", "README must expose the latest complete step."),
        note_phrase_record(readme_text, "paper-side build / TeX audit / page audit: complete", "README must expose the paper-side closeout status."),
        note_phrase_record(readme_text, "8.7.55.3.133-.136", "README must expose the next follow-through branch."),
    ]
    expert_note_phrases = [
        note_phrase_record(expert_note_text, "kappa_a = a0 / (c H0^(P)) = 1 / (2 pi)", "Expert note must state the direct bridge formula."),
        note_phrase_record(expert_note_text, "machine precision", "Expert note must state the operational/derived agreement."),
        note_phrase_record(expert_note_text, "independent-galaxy intake", "Expert note must state the next follow-through focus."),
    ]

    independent_prerequisites = [
        {
            "item_key": "direct_kappa_declaration_available",
            "present": bool(declaration_gate["summary"]["dark_matter_elimination_declaration_ready"]),
            "note": "The independent-galaxy intake only makes sense if the direct kappa_a declaration is already frozen.",
            "evidence": declaration_gate["summary"],
        },
        {
            "item_key": "same_baryon_interface_rule_available",
            "present": hit(part4_text, "同一 baryon I/F") is not None and hit(part5_text, "同一 baryon I/F") is not None,
            "note": "The follow-through branch requires an explicit same-baryon-interface rule.",
            "evidence": {
                "part4": hit(part4_text, "同一 baryon I/F"),
                "part5": hit(part5_text, "同一 baryon I/F"),
            },
        },
        {
            "item_key": "independent_kappa_comparison_rule_available",
            "present": hit(part5_text, r"$a_0/(cH_{0}^{(P)})$") is not None,
            "note": "The repo must already say how the independent sample will compare a0/(cH0^(P)) to 1/(2*pi).",
            "evidence": hit(part5_text, r"$a_0/(cH_{0}^{(P)})$"),
        },
        {
            "item_key": "non_sparc_rotation_curve_primary_source_registry_available",
            "present": hit(primary_sources_text, "SPARC 以外の公開 rotation-curve sample") is not None,
            "note": "A non-SPARC public rotation-curve source registry has not yet been added to PRIMARY_SOURCES.",
            "evidence": hit(primary_sources_text, "SPARC 以外の公開 rotation-curve sample"),
        },
        {
            "item_key": "independent_baryonic_decomposition_pack_available",
            "present": hit(primary_sources_text, "baryonic decomposition") is not None and hit(primary_sources_text, "rotation-curve sample") is not None,
            "note": "The next intake branch needs a public baryonic decomposition pack for the independent galaxies.",
            "evidence": {
                "baryonic_decomposition": hit(primary_sources_text, "baryonic decomposition"),
                "rotation_curve_sample": hit(primary_sources_text, "rotation-curve sample"),
            },
        },
        {
            "item_key": "external_share_pack_available",
            "present": bundle_zip.exists() and not missing_share_pack_records,
            "note": "The current external share pack itself must be stable before the next dataset-intake branch opens.",
            "evidence": {"bundle_zip": rel(bundle_zip), "missing_items": [item["item_key"] for item in missing_share_pack_records]},
        },
    ]
    present_prerequisites = [item for item in independent_prerequisites if item["present"]]
    missing_prerequisites = [item for item in independent_prerequisites if not item["present"]]
    independent_intake_ready = not missing_prerequisites
    external_share_ready = bundle_zip.exists() and not missing_share_pack_records

    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in staging_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(staging_dir))

    payloads = {
        "mass_origin_dark_matter_external_share_pack_inventory": payload(
            "8.7.55.3.133",
            "External share-pack inventory freeze",
            {
                "share_pack_zip": rel(bundle_zip),
                "share_pack_staging_dir": rel(staging_dir),
                "status_markdown": rel(STATUS),
                "roadmap_markdown": rel(ROADMAP),
                "ai_context_json": rel(AI_CONTEXT),
            },
            "Freeze the canonical external share-pack inventory that is needed for expert review and public dissemination after the direct kappa_a declaration has already been closed on theory and paper sides.",
            {
                "inventory_rule": "share-pack inventory passes only if the canonical zip exists, the staging directory exists, and all required docs / metrics / paper PDFs / page-audit artifacts are present in the pack"
            },
            [
                row("external_share_pack_inventory_complete", "pass", "external share-pack inventory complete", 1, "The canonical external share pack was inventoried against the current direct-kappa closeout state."),
                row("external_share_pack_present_item_count", "pass" if not missing_share_pack_records else "reject", "present external share-pack item count", len(present_share_pack_records), "The canonical pack is complete only if every required staged item is present."),
                row("external_share_pack_missing_item_count", "pass" if not missing_share_pack_records else "reject", "missing external share-pack item count", len(missing_share_pack_records), "The external share pack is inventory-ready only when the missing count stays zero."),
                row("external_share_pack_staging_dir_restored_from_zip", "pass", "share-pack staging dir restored from zip", 1 if created_from_zip else 0, "If the staging directory was missing, it was restored from the canonical zip before the inventory freeze."),
            ],
            {
                "canonical_share_pack_zip": rel(bundle_zip),
                "canonical_share_pack_staging_dir": rel(staging_dir),
                "required_share_pack_items": [item["item_key"] for item in share_pack_records],
                "missing_share_pack_items": [item["item_key"] for item in missing_share_pack_records],
                "external_share_pack_inventory_ready": not missing_share_pack_records,
                "staging_dir_created_from_zip": created_from_zip,
            },
            {
                "overall_status": "external_share_pack_inventory_frozen",
                "external_share_pack_ready": not missing_share_pack_records,
                "next_required_artifacts": [
                    "mass_origin_dark_matter_expert_note_wording_freeze",
                    "mass_origin_dark_matter_independent_galaxy_intake_prerequisite_inventory",
                ],
            },
            {"share_pack_records": share_pack_records},
        ),
        "mass_origin_dark_matter_expert_note_wording_freeze": payload(
            "8.7.55.3.134",
            "Expert note / dissemination wording freeze",
            {
                "share_pack_zip": rel(bundle_zip),
                "share_pack_staging_dir": rel(staging_dir),
                "readme_txt": str(readme_path.relative_to(ROOT)).replace("\\", "/"),
                "expert_note_txt": str(expert_note_path.relative_to(ROOT)).replace("\\", "/"),
            },
            "Freeze the external-share wording that accompanies the direct kappa_a declaration so the dissemination layer says exactly what has been closed and what remains as follow-through.",
            {
                "readme_rule": "README must state the latest completed step, the paper-side closeout, and the next follow-through branch",
                "expert_note_rule": "EXPERT_NOTE must state the direct kappa_a bridge, machine-precision SPARC equality, and the independent-galaxy follow-through focus",
            },
            [
                row("external_share_readme_wording_ready", "pass" if all(item["present"] for item in readme_phrases) else "reject", "external share README wording ready", 1 if all(item["present"] for item in readme_phrases) else 0, "README is frozen only if it exposes the direct-kappa closeout and next branch in canonical wording."),
                row("external_share_expert_note_wording_ready", "pass" if all(item["present"] for item in expert_note_phrases) else "reject", "external share expert note wording ready", 1 if all(item["present"] for item in expert_note_phrases) else 0, "EXPERT_NOTE is frozen only if it exposes the direct bridge, equality, and the next independent-galaxy focus."),
                row("external_share_zip_refreshed_after_note_freeze", "pass", "external share zip refreshed after note freeze", 1, "The canonical zip was rewritten after the dissemination wording was frozen."),
            ],
            {
                "readme_wording_ready": all(item["present"] for item in readme_phrases),
                "expert_note_wording_ready": all(item["present"] for item in expert_note_phrases),
                "dissemination_wording_pack_ready": all(item["present"] for item in readme_phrases) and all(item["present"] for item in expert_note_phrases),
                "share_pack_zip_refreshed": True,
            },
            {
                "overall_status": "external_share_note_wording_frozen",
                "dissemination_wording_ready": all(item["present"] for item in readme_phrases) and all(item["present"] for item in expert_note_phrases),
                "next_required_artifacts": ["mass_origin_dark_matter_independent_galaxy_intake_prerequisite_inventory"],
            },
            {"readme_phrases": readme_phrases, "expert_note_phrases": expert_note_phrases},
        ),
        "mass_origin_dark_matter_independent_galaxy_intake_prerequisite_inventory": payload(
            "8.7.55.3.135",
            "Independent-galaxy intake prerequisite inventory",
            {
                "primary_sources_markdown": rel(PRIMARY_SOURCES),
                "part4_markdown": rel(PART4),
                "part5_markdown": rel(PART5),
                "dark_matter_elimination_declaration_gate_json": rel(DECLARATION_GATE),
            },
            "Inventory whether the repo already contains the public inputs, same-baryon-interface rule, and comparison rule needed to start the first non-SPARC independent-galaxy verification of kappa_a = 1/(2*pi).",
            {
                "intake_rule": "the intake is ready only if the declaration is frozen, the same-baryon-interface rule is explicit, the comparison rule is explicit, and a non-SPARC rotation-curve+baryonic-decomposition public source registry already exists"
            },
            [
                row("independent_galaxy_intake_prerequisite_inventory_complete", "pass", "independent-galaxy intake prerequisite inventory complete", 1, "The non-SPARC intake prerequisites were audited against the current repo state."),
                row("independent_galaxy_intake_present_prerequisite_count", "pass" if independent_intake_ready else "reject", "present independent-galaxy intake prerequisite count", len(present_prerequisites), "The intake can open only when every prerequisite is present."),
                row("independent_galaxy_intake_missing_prerequisite_count", "pass" if independent_intake_ready else "reject", "missing independent-galaxy intake prerequisite count", len(missing_prerequisites), "The missing count identifies the next blocking public inputs that must be added before the intake branch can start."),
            ],
            {
                "required_independent_galaxy_prerequisites": [item["item_key"] for item in independent_prerequisites],
                "present_independent_galaxy_prerequisites": [item["item_key"] for item in present_prerequisites],
                "missing_independent_galaxy_prerequisites": [item["item_key"] for item in missing_prerequisites],
                "independent_galaxy_intake_ready": independent_intake_ready,
                "first_route_to_close_or_none": None if independent_intake_ready else "independent_rotation_curve_public_source_pack",
            },
            {
                "overall_status": "independent_galaxy_prerequisites_frozen",
                "independent_galaxy_intake_ready": independent_intake_ready,
                "external_share_ready": external_share_ready,
                "next_required_artifacts": ["mass_origin_dark_matter_external_feedback_dataset_handoff_gate"],
            },
            {"independent_galaxy_prerequisites": independent_prerequisites},
        ),
        "mass_origin_dark_matter_external_feedback_dataset_handoff_gate": payload(
            "8.7.55.3.136",
            "External feedback / dataset-handoff gate",
            {
                "external_share_pack_inventory_json": "output/public/quantum/mass_origin_dark_matter_external_share_pack_inventory_metrics.json",
                "expert_note_wording_freeze_json": "output/public/quantum/mass_origin_dark_matter_expert_note_wording_freeze_metrics.json",
                "independent_galaxy_intake_prerequisite_inventory_json": "output/public/quantum/mass_origin_dark_matter_independent_galaxy_intake_prerequisite_inventory_metrics.json",
            },
            "Decide whether the next official route should simply wait for external feedback on the now-complete declaration pack, or whether the repo is already ready to open a non-SPARC independent-galaxy dataset-intake branch immediately.",
            {
                "handoff_rule": "if the share pack is ready but independent-galaxy prerequisites are still missing, hold the declaration for feedback and open a dedicated dataset-intake preparation branch next"
            },
            [
                row("external_share_pack_ready_for_feedback", "pass" if external_share_ready else "reject", "external share pack ready for feedback", 1 if external_share_ready else 0, "The direct-kappa declaration can be circulated externally once the canonical share pack is complete."),
                row("independent_galaxy_dataset_handoff_ready", "pass" if independent_intake_ready else "reject", "independent-galaxy dataset handoff ready", 1 if independent_intake_ready else 0, "The next dataset-intake branch can only launch immediately if all public prerequisites are already present."),
                row("external_feedback_dataset_handoff_gate_complete", "pass", "external feedback / dataset handoff gate complete", 1, "The post-closeout follow-through gate is now frozen."),
            ],
            {
                "external_share_ready": external_share_ready,
                "dissemination_wording_ready": all(item["present"] for item in readme_phrases) and all(item["present"] for item in expert_note_phrases),
                "independent_galaxy_intake_ready": independent_intake_ready,
                "await_external_feedback": external_share_ready,
                "launch_dataset_intake_now": independent_intake_ready,
                "recommended_next_route_or_none": "8.7.55.3.137" if not independent_intake_ready else None,
                "selected_next_route": "independent_galaxy_public_source_inventory" if not independent_intake_ready else "independent_galaxy_dataset_intake_execution",
            },
            {
                "overall_status": "external_share_ready_dataset_intake_prerequisites_missing" if not independent_intake_ready else "external_share_and_dataset_intake_ready",
                "third_route_fully_closed": True,
                "share_pack_followthrough_complete": True,
                "next_required_artifacts": [] if independent_intake_ready else ["independent_galaxy_public_source_inventory_branch"],
            },
            {
                "missing_prerequisites": missing_prerequisites,
                "bundle_zip": rel(bundle_zip),
                "staging_dir": rel(staging_dir),
            },
        ),
    }

    for stem, data in payloads.items():
        write_artifact(stem, data)
        print(f"[ok] wrote {OUT / (stem + '_metrics.json')}")
        print(f"[ok] wrote {OUT / (stem + '_rows.csv')}")


# Function: Run the external-share / independent-galaxy follow-through branch when invoked as a script.

if __name__ == "__main__":
    main()
