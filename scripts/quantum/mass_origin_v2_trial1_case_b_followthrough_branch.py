#!/usr/bin/env python3
"""
Generate Trial-1 Case-B follow-through artifacts for 8.7.56.141-.144.

The VEV pivot already answered the physical question: under the current canon,
the transverse fluctuation of P_mu remains massive and does not become a
massless photon. The remaining work is therefore synchronization, not another
physics retry. This branch:

1. inventories the canonical paper-side / control-doc / share-pack targets that
   must reflect the Case-B outcome,
2. freezes the Part III-A / Part V wording pack that states the honest partial
   closeout,
3. freezes the scope declaration gate for Trial-1 / Trial-2 under the current
   canon, and
4. refreshes the expert-share bundle and freezes the next official route as a
   future-canon-delta inventory rather than a Trial-2 launch.
"""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PRIVATE_QUANTUM = ROOT / "output" / "private" / "quantum"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
VEV_MASS = OUT / "mass_origin_v2_transverse_mode_effective_mass_evaluation_metrics.json"
VEV_GATE = OUT / "mass_origin_v2_trial1_vev_pivot_reopened_declaration_gate_metrics.json"
CASE_B_GATE = OUT / "mass_origin_v2_trial1_case_b_honest_partial_closeout_v1_1_confirmation_gate_metrics.json"
FOLLOWTHROUGH = OUT / "mass_origin_v2_trial1_case_b_followthrough_route_contract_metrics.json"


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: return a compact UTC timestamp for filenames.

def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# Function: abort if a required path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: load a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: load a UTF-8 text file.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: convert an absolute path into a repository-relative path.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard result row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard payload object.

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


# Function: save a JSON artifact and its row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: return the most recent expert-review bundle zip or None.

def latest_bundle_zip() -> Path | None:
    candidates = sorted(PRIVATE_QUANTUM.glob("expert_review_bundle_*.zip"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        return None

    return candidates[-1]


# Function: write canonical share-pack note files for the Case-B state.

def write_case_b_notes(bundle_dir: Path) -> tuple[Path, Path]:
    readme = bundle_dir / "README.txt"
    note = bundle_dir / "EXPERT_NOTE.txt"
    generated = now_iso()
    readme.write_text(
        "waveP expert review bundle\n"
        f"Generated: {generated}\n\n"
        "Scope\n"
        "- Current Phase 8 status after completing 8.7.56.140.\n"
        "- Trial-1 VEV pivot is now physically closed on Case B.\n"
        "- The current canon keeps the transverse P_mu mode massive, so photon derivation from P_mu alone is not available.\n\n"
        "Current state\n"
        "- Latest complete step: 8.7.56.140\n"
        "- Trial-1 pass level: honest_partial_closeout_case_b_transverse_massive\n"
        "- Trial-2 status: hold\n"
        "- Next official branch: 8.7.56.141-.144 Trial-1 Case-B follow-through\n",
        encoding="utf-8",
    )
    note.write_text(
        "Expert note\n\n"
        "The VEV pivot has now answered the Trial-1 photon question on the current canon.\n\n"
        "Frozen result\n"
        "- The mexican-hat contribution to the transverse quadratic mass vanishes at the anchored VEV.\n"
        "- The explicit Proca/Stueckelberg term survives, so the transverse mode remains massive:\n"
        "  m_T^2 = m_P^2 = 2 lambda v^2 / Z_P.\n"
        "- Therefore the present canon does not derive a massless photon from P_mu alone.\n\n"
        "Consequences\n"
        "- independent L_EM retention is physically required under the current canon.\n"
        "- v1.1 judgment is preserved.\n"
        "- Trial-1 is an honest partial closeout, not a full derivation.\n"
        "- Trial-2 remains on hold pending a future canon change that can truly remove the transverse Proca mass.\n",
        encoding="utf-8",
    )
    return readme, note


# Function: create a refreshed expert-review bundle for the current Case-B state.

def create_case_b_bundle(files_to_sync: list[Path]) -> tuple[Path, Path]:
    stamp = now_stamp()
    bundle_dir = PRIVATE_QUANTUM / f"expert_review_bundle_{stamp}"
    bundle_zip = PRIVATE_QUANTUM / f"expert_review_bundle_{stamp}.zip"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    base_zip = latest_bundle_zip()
    if base_zip is not None:
        with zipfile.ZipFile(base_zip, "r") as archive:
            archive.extractall(bundle_dir)

    for source in files_to_sync:
        target = bundle_dir / source.name
        shutil.copy2(source, target)

    write_case_b_notes(bundle_dir)

    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(bundle_dir))

    return bundle_dir, bundle_zip


# Function: build an inventory record for a wording target.

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: execute the Case-B follow-through branch and freeze the next route contract.

def main() -> None:
    for path in (PART3A, PART5, STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, VEV_MASS, VEV_GATE, CASE_B_GATE, FOLLOWTHROUGH):
        req(path)

    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    ai_context = read_json(AI_CONTEXT)
    vev_mass = read_json(VEV_MASS)
    vev_gate = read_json(VEV_GATE)
    case_b_gate = read_json(CASE_B_GATE)
    followthrough = read_json(FOLLOWTHROUGH)

    common_inputs = {
        "part3a_markdown": rel(PART3A),
        "part5_markdown": rel(PART5),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "work_history_recent_markdown": rel(WORK_HISTORY_RECENT),
        "vev_mass_metrics_json": rel(VEV_MASS),
        "vev_gate_metrics_json": rel(VEV_GATE),
        "case_b_gate_metrics_json": rel(CASE_B_GATE),
        "case_b_followthrough_metrics_json": rel(FOLLOWTHROUGH),
    }

    inventory_targets = [
        target_record(
            "part3a_case_b_mass_formula",
            PART3A,
            part3a_text,
            "m_T^2=m_P^2=2\\lambda v^2/Z_P\\neq0",
            "Part III-A must expose the explicit Case-B transverse mass formula.",
        ),
        target_record(
            "part3a_case_b_partial_closeout",
            PART3A,
            part3a_text,
            "honest partial closeout",
            "Part III-A must state that Trial-1 closes as an honest partial closeout.",
        ),
        target_record(
            "part3a_case_b_trial2_hold",
            PART3A,
            part3a_text,
            "Trial-2 は hold を継続する。",
            "Part III-A must state that Trial-2 remains on hold under Case B.",
        ),
        target_record(
            "part5_case_b_section",
            PART5,
            part5_text,
            "### 3.2 v2.0 試練1：VEV transverse-photon challenge",
            "Part V must contain the Trial-1 Case-B challenge section.",
        ),
        target_record(
            "part5_case_b_mass_formula",
            PART5,
            part5_text,
            "m_T^2=m_P^2=2\\lambda v^2/Z_P\\neq0",
            "Part V must expose the explicit Case-B mass formula.",
        ),
        target_record(
            "part5_case_b_trial2_hold",
            PART5,
            part5_text,
            "Trial-2 を unlock する。",
            "Part V must describe the reopen condition for Trial-2.",
        ),
        target_record(
            "status_next_step_case_b_followthrough",
            STATUS,
            status_text,
            "current official next step は `8.7.56.141`",
            "STATUS must reflect the Case-B follow-through branch as the current official next step.",
        ),
    ]

    present_targets = [item for item in inventory_targets if item["present"]]
    missing_targets = [item for item in inventory_targets if not item["present"]]
    part3a_ready = all(item["present"] for item in inventory_targets if item["file_key"].startswith("part3a_"))
    part5_ready = all(item["present"] for item in inventory_targets if item["file_key"].startswith("part5_"))
    status_ready = all(item["present"] for item in inventory_targets if item["file_key"].startswith("status_"))
    ai_context_ready = "8.7.56.141" in ai_context["current_step"]

    inventory = payload(
        "8.7.56.141",
        "Trial-1 Case-B paper-side sync inventory",
        common_inputs,
        "Inventory the canonical paper-side, control-doc, and share-pack targets that must now reflect the Case-B VEV-pivot outcome.",
        {
            "inventory_rule": "inventory passes only if Part III-A, Part V, STATUS, and AI_CONTEXT already expose the Case-B wording targets required by the frozen VEV pivot outcome",
        },
        [
            row(
                "trial1_case_b_sync_inventory_complete",
                "pass",
                "Trial-1 Case-B sync inventory complete",
                1,
                "The Case-B sync inventory was executed against the updated sources.",
            ),
            row(
                "trial1_case_b_sync_present_target_count",
                "pass" if not missing_targets else "reject",
                "present Case-B sync target count",
                len(present_targets),
                "The sync inventory closes only if every required target is present.",
            ),
            row(
                "trial1_case_b_sync_missing_target_count",
                "pass" if not missing_targets else "reject",
                "missing Case-B sync target count",
                len(missing_targets),
                "The missing count identifies any remaining wording drift after the Case-B update.",
            ),
        ],
        {
            "required_sync_targets": [item["file_key"] for item in inventory_targets] + ["ai_context_case_b_next_step"],
            "present_sync_targets": [item["file_key"] for item in present_targets] + (["ai_context_case_b_next_step"] if ai_context_ready else []),
            "missing_sync_targets": [item["file_key"] for item in missing_targets] + ([] if ai_context_ready else ["ai_context_case_b_next_step"]),
            "part3a_case_b_wording_ready": part3a_ready,
            "part5_case_b_wording_ready": part5_ready,
            "status_case_b_branch_ready": status_ready,
            "ai_context_case_b_branch_ready": ai_context_ready,
            "first_route_to_close_or_none": None if not missing_targets and ai_context_ready else "case_b_wording_target_missing",
        },
        {
            "overall_status": "trial1_case_b_sync_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_142": True,
            "next_required_artifacts": [
                "trial1_case_b_part3a_part5_wording_freeze",
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    wording_freeze = payload(
        "8.7.56.142",
        "Part III-A / Part V Case-B wording freeze",
        common_inputs,
        "Freeze the paper-side wording that states the VEV-pivot Case-B conclusion, the retained independent electromagnetic sector, and the continued Trial-2 hold.",
        {
            "part3a_rule": "Part III-A must state m_T^2 = m_P^2 != 0, honest partial closeout, and Trial-2 hold.",
            "part5_rule": "Part V must expose the Trial-1 challenge section with the current-canon Case-B state and the reopen condition.",
        },
        [
            row(
                "trial1_case_b_part3a_wording_ready",
                "pass" if part3a_ready else "reject",
                "Part III-A Case-B wording ready",
                1 if part3a_ready else 0,
                "Part III-A is ready only if all Case-B wording targets are present.",
            ),
            row(
                "trial1_case_b_part5_wording_ready",
                "pass" if part5_ready else "reject",
                "Part V Case-B wording ready",
                1 if part5_ready else 0,
                "Part V is ready only if the challenge section and reopen condition are present.",
            ),
            row(
                "trial1_case_b_control_doc_wording_ready",
                "pass" if status_ready and ai_context_ready else "reject",
                "Case-B control-doc wording ready",
                1 if status_ready and ai_context_ready else 0,
                "STATUS and AI_CONTEXT must already track the Case-B follow-through state.",
            ),
        ],
        {
            "part3a_case_b_wording_ready": part3a_ready,
            "part5_case_b_wording_ready": part5_ready,
            "control_doc_case_b_wording_ready": status_ready and ai_context_ready,
            "paper_side_case_b_wording_pack_ready": part3a_ready and part5_ready and status_ready and ai_context_ready,
        },
        {
            "overall_status": "trial1_case_b_wording_pack_frozen" if part3a_ready and part5_ready and status_ready and ai_context_ready else "trial1_case_b_wording_pack_missing",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_143": True,
            "next_required_artifacts": [
                "trial1_case_b_scope_declaration_gate",
            ],
        },
        {
            "part3a_case_b_mass_line": hit(part3a_text, "m_T^2=m_P^2=2\\lambda v^2/Z_P\\neq0"),
            "part3a_trial2_hold_line": hit(part3a_text, "Trial-2 は hold を継続する。"),
            "part5_section_line": hit(part5_text, "### 3.2 v2.0 試練1：VEV transverse-photon challenge"),
            "part5_reopen_line": hit(part5_text, "Trial-2 を unlock する。"),
        },
    )

    scope_gate = payload(
        "8.7.56.143",
        "Trial-1 Case-B scope declaration gate",
        common_inputs,
        "Freeze the current-canon scope declaration after the Case-B sync: Trial-1 is an honest partial closeout, v2.0 minimum condition is unmet, and Trial-2 remains on hold.",
        {
            "scope_rule": "current canon closes Trial-1 as Case B if m_T^2 != 0 and the independent electromagnetic sector is retained.",
            "minimum_condition_rule": "v2.0 minimum condition remains unmet until Trial-1 derives a photon and Trial-2 unlocks.",
            "trial2_rule": "Trial-2 remains on hold until a future canon change creates a genuine massless photon route.",
        },
        [
            row(
                "trial1_case_b_scope_declared",
                "pass",
                "Trial-1 Case-B scope declared",
                1,
                "The current-canon scope is now explicit: honest partial closeout, not full derivation.",
            ),
            row(
                "trial1_case_b_v2_minimum_condition_met",
                "fail",
                "v2.0 minimum condition met under current canon",
                0,
                "The current canon does not satisfy the Trial-1 photon-derivation condition and therefore does not unlock Trial-2.",
            ),
            row(
                "trial1_case_b_trial2_hold_retained",
                "pass",
                "Trial-2 hold retained under current canon",
                1,
                "The Case-B result keeps Trial-2 on hold rather than reopening it prematurely.",
            ),
        ],
        {
            "trial1_scope_declared_as_honest_partial_closeout": True,
            "v2_minimum_condition_satisfied_under_current_canon": False,
            "trial2_hold_retained": True,
            "ready_for_future_canon_delta_inventory": True,
            "recommended_next_route_or_none": "8.7.56.144",
        },
        {
            "overall_status": "trial1_case_b_scope_declaration_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_144": True,
            "next_required_artifacts": [
                "trial1_case_b_share_pack_followthrough_route_contract",
            ],
        },
        {
            "vev_mass_summary": vev_mass["summary"],
            "vev_gate_summary": vev_gate["summary"],
            "case_b_gate_summary": case_b_gate["summary"],
            "followthrough_summary": followthrough["summary"],
        },
    )

    bundle_files = [
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PART3A,
        PART5,
        VEV_MASS,
        VEV_GATE,
        CASE_B_GATE,
        FOLLOWTHROUGH,
    ]
    bundle_dir, bundle_zip = create_case_b_bundle(bundle_files)
    readme_text = read_text(bundle_dir / "README.txt")
    note_text = read_text(bundle_dir / "EXPERT_NOTE.txt")

    followthrough_contract = payload(
        "8.7.56.144",
        "Trial-1 Case-B share-pack / reserve follow-through route contract",
        common_inputs,
        "Refresh the expert-share bundle for the Case-B state and freeze the next official route as a future-canon-delta inventory instead of a Trial-2 launch.",
        {
            "selected_followthrough_route": "trial1_case_b_future_canon_delta_inventory",
            "missing_v2_artifact": "trial1_case_b_future_canon_delta_registry",
            "followthrough_rule": "The physics question is already answered; the next route catalogs the minimal canon deltas that would be required to reopen Trial-1 honestly.",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold while the future-canon-delta inventory is prepared.",
        },
        [
            row(
                "trial1_case_b_share_pack_bundle_refreshed",
                "pass",
                "Trial-1 Case-B share-pack bundle refreshed",
                1,
                "A refreshed expert-review bundle was generated for the current Case-B state.",
            ),
            row(
                "trial1_case_b_share_pack_readme_ready",
                "pass" if "Latest complete step: 8.7.56.140" in readme_text else "reject",
                "Case-B share-pack README ready",
                1 if "Latest complete step: 8.7.56.140" in readme_text else 0,
                "The refreshed README must expose the latest complete step and the Case-B state.",
            ),
            row(
                "trial1_case_b_share_pack_expert_note_ready",
                "pass" if "m_T^2 = m_P^2 = 2 lambda v^2 / Z_P." in note_text else "reject",
                "Case-B share-pack expert note ready",
                1 if "m_T^2 = m_P^2 = 2 lambda v^2 / Z_P." in note_text else 0,
                "The refreshed expert note must expose the explicit Case-B transverse mass formula.",
            ),
            row(
                "trial1_case_b_next_route_shifted_to_future_canon_delta_inventory",
                "pass",
                "next route shifted to future-canon-delta inventory",
                1,
                "The next official work is a future-canon-delta registry, not another immediate physics retry.",
            ),
        ],
        {
            "case_b_share_pack_bundle_dir": rel(bundle_dir),
            "case_b_share_pack_bundle_zip": rel(bundle_zip),
            "selected_followthrough_route": "trial1_case_b_future_canon_delta_inventory",
            "missing_v2_artifact": "trial1_case_b_future_canon_delta_registry",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.145",
        },
        {
            "overall_status": "trial1_case_b_share_pack_followthrough_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "trial1_case_b_future_canon_delta_inventory",
                "trial1_case_b_future_canon_delta_admissibility_audit",
                "trial1_case_b_future_canon_delta_declaration_gate",
            ],
        },
        {
            "bundle_readme_line": hit(readme_text, "Latest complete step: 8.7.56.140"),
            "bundle_note_line": hit(note_text, "m_T^2 = m_P^2 = 2 lambda v^2 / Z_P."),
            "scope_gate_summary": scope_gate["summary"],
            "case_b_gate_summary": case_b_gate["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial1_case_b_paper_sync_inventory", inventory)
    write_artifact("mass_origin_v2_trial1_case_b_part3a_part5_wording_freeze", wording_freeze)
    write_artifact("mass_origin_v2_trial1_case_b_scope_declaration_gate", scope_gate)
    write_artifact("mass_origin_v2_trial1_case_b_share_pack_followthrough_route_contract", followthrough_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial1_case_b_paper_sync_inventory_metrics.json")
    print(" - mass_origin_v2_trial1_case_b_part3a_part5_wording_freeze_metrics.json")
    print(" - mass_origin_v2_trial1_case_b_scope_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial1_case_b_share_pack_followthrough_route_contract_metrics.json")
    print(f" - {bundle_zip}")


# Function: run the follow-through branch from the command line.

if __name__ == "__main__":
    main()
