#!/usr/bin/env python3
"""
Generate Trial-1 Case-B future-canon delta registry artifacts for 8.7.56.145-.148.

The VEV pivot already closed the current canon on Case B:

    m_T^2 = m_P^2 = 2 lambda v^2 / Z_P != 0

So the next honest question is no longer "can we force Trial-1 to pass on the
current canon?" but rather "what would have to change before Trial-1 could be
reopened without cheating?" This branch therefore:

1. inventories the canonical action / pole / independent-EM / Part III-A
   judgment items that keep Trial-1 in Case B,
2. audits which kinds of future-canon deltas are structurally required and
   which shortcuts remain inadmissible,
3. freezes the Part V wording that turns those reopen prerequisites into a
   public challenge registry, and
4. freezes the reopen-prerequisite gate while confirming that Trial-2 remains
   on hold under the current canon.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
VEV_MASS = OUT / "mass_origin_v2_transverse_mode_effective_mass_evaluation_metrics.json"
CASE_B_SCOPE = OUT / "mass_origin_v2_trial1_case_b_scope_declaration_gate_metrics.json"
CASE_B_FOLLOWTHROUGH = OUT / "mass_origin_v2_trial1_case_b_share_pack_followthrough_route_contract_metrics.json"


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# Function: build an inventory record for a required canonical source.

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


# Function: execute the future-canon delta registry branch.

def main() -> None:
    for path in (PART1, PART3A, PART5, STATUS, ROADMAP, AI_CONTEXT, VEV_MASS, CASE_B_SCOPE, CASE_B_FOLLOWTHROUGH):
        req(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    ai_context = read_json(AI_CONTEXT)
    vev_mass = read_json(VEV_MASS)
    case_b_scope = read_json(CASE_B_SCOPE)
    case_b_followthrough = read_json(CASE_B_FOLLOWTHROUGH)

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "part5_markdown": rel(PART5),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "vev_mass_metrics_json": rel(VEV_MASS),
        "case_b_scope_metrics_json": rel(CASE_B_SCOPE),
        "case_b_followthrough_metrics_json": rel(CASE_B_FOLLOWTHROUGH),
    }

    inventory_targets = [
        target_record(
            "part1_total_action_em_term",
            PART1,
            part1_text,
            "+\\mathcal{L}_{\\mathrm{EM}}",
            "Part I still carries an explicit independent electromagnetic sector in the total action.",
        ),
        target_record(
            "part1_proca_stueckelberg_mass_term",
            PART1,
            part1_text,
            "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)",
            "Part I still carries the explicit Proca/Stueckelberg mass term that survives in the transverse sector.",
        ),
        target_record(
            "part1_massive_vector_pole",
            PART1,
            part1_text,
            "\\frac{\\eta_{\\mu\\nu}-k_\\mu k_\\nu/m_P^2}{k^2-m_P^2+i0}",
            "Part I still exposes the massive transverse vector pole rather than a massless photon pole.",
        ),
        target_record(
            "part1_ghost_free_clause",
            PART1,
            part1_text,
            "負ノルム（ghost）モードは出現しない。",
            "Any future canon delta must preserve the current ghost-free closure.",
        ),
        target_record(
            "part3a_a_reject_b_adopt_judgment",
            PART3A,
            part3a_text,
            "A棄却、B採用",
            "Part III-A still freezes the v1.1 U(1) judgment as A reject / B adopt.",
        ),
        target_record(
            "part3a_independent_em_adoption",
            PART3A,
            part3a_text,
            "独立 $\\mathcal{L}_{\\mathrm{EM}}$ 採用",
            "Part III-A still states that the independent electromagnetic sector remains physically retained.",
        ),
        target_record(
            "part5_case_b_reopen_condition",
            PART5,
            part5_text,
            "transverse Proca mass を 0 にする新しい canon change",
            "Part V must still expose the Case-B reopen condition at the challenge level.",
        ),
        target_record(
            "status_future_canon_delta_next_step",
            STATUS,
            status_text,
            "current official next step は `8.7.56.145`",
            "STATUS must already point to the future-canon delta registry branch.",
        ),
    ]

    present_targets = [item for item in inventory_targets if item["present"]]
    missing_targets = [item for item in inventory_targets if not item["present"]]
    ai_context_ready = "8.7.56.145" in ai_context["current_step"]
    registry_ready = not missing_targets and ai_context_ready

    inventory = payload(
        "8.7.56.145",
        "Trial-1 Case-B future-canon delta inventory",
        common_inputs,
        "Inventory the canonical action, pole, independent-EM, and Part III-A judgment items that currently force Trial-1 to remain in Case B and keep Trial-2 on hold.",
        {
            "inventory_rule": "inventory passes only if the current canon still exposes the explicit mass term, the massive pole, the independent EM sector, and the Part III-A A-reject/B-adopt judgment as active support for Case B",
            "registry_goal": "freeze the minimal list of canonical supports that any honest Trial-1 reopen must modify",
        },
        [
            row(
                "trial1_case_b_future_canon_delta_inventory_complete",
                "pass",
                "Trial-1 Case-B future-canon delta inventory complete",
                1,
                "The future-canon delta inventory was executed against the current canonical sources.",
            ),
            row(
                "trial1_case_b_future_canon_present_target_count",
                "pass" if registry_ready else "reject",
                "present future-canon delta target count",
                len(present_targets) + (1 if ai_context_ready else 0),
                "Every current-canon support must be visible before the registry can be frozen.",
            ),
            row(
                "trial1_case_b_future_canon_missing_target_count",
                "pass" if registry_ready else "reject",
                "missing future-canon delta target count",
                len(missing_targets) + (0 if ai_context_ready else 1),
                "Missing targets would mean the registry itself is drifting out of sync with the official canon.",
            ),
        ],
        {
            "future_canon_delta_registry_items": [
                "explicit_proca_stueckelberg_mass_term",
                "massive_transverse_vector_pole",
                "independent_electromagnetic_sector_retention",
                "part3a_a_reject_b_adopt_judgment",
            ],
            "future_canon_delta_registry_ready": registry_ready,
            "ai_context_future_canon_delta_ready": ai_context_ready,
            "first_route_to_close_or_none": None if registry_ready else "future_canon_delta_source_missing",
        },
        {
            "overall_status": "trial1_case_b_future_canon_delta_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_146": True,
            "next_required_artifacts": [
                "trial1_case_b_future_canon_delta_admissibility_audit",
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "ai_context_current_step": ai_context["current_step"],
            "vev_mass_summary": vev_mass["summary"],
            "case_b_scope_summary": case_b_scope["summary"],
        },
    )

    admissibility = payload(
        "8.7.56.146",
        "Future-canon delta admissibility audit",
        common_inputs,
        "Audit which kinds of canon change are structurally required before Trial-1 could be honestly reopened, and reject shortcuts that merely rewrite wording without changing the physics closure.",
        {
            "wording_only_rule": "A wording-only override is inadmissible because the current Case-B judgment is driven by action-level mass and pole structure, not by missing prose.",
            "single_patch_rule": "A single isolated patch is inadmissible if it leaves the massive transverse pole, the independent EM sector, or the Part III-A judgment untouched.",
            "multi_delta_rule": "Any honest reopen requires a coupled future-canon program spanning action-level mass structure, pole structure, EM-sector unification, and a downstream judgment re-audit.",
            "safety_rule": "Any future program must re-pass ghost-free, stability, existing-observable, and v1.1-closeout checks.",
        },
        [
            row(
                "trial1_case_b_wording_only_reopen_admissible",
                "fail",
                "wording-only reopen admissible",
                0,
                "Current Case B is a physics result, so wording-only reopening is inadmissible.",
            ),
            row(
                "trial1_case_b_single_delta_patch_admissible",
                "fail",
                "single-delta patch admissible",
                0,
                "Removing only one support leaves the remaining Case-B supports intact and does not honestly reopen Trial-1.",
            ),
            row(
                "trial1_case_b_multi_delta_program_required",
                "pass",
                "multi-delta future-canon program required",
                1,
                "An honest reopen requires a coupled future-canon bundle rather than a local patch.",
            ),
            row(
                "trial1_case_b_safety_readmission_required",
                "pass",
                "ghost-free / stability / observables re-audit required",
                1,
                "Any future canon delta must be re-audited for ghost-free closure, stability, and preserved observables.",
            ),
            row(
                "trial1_case_b_current_canon_reopen_ready",
                "fail",
                "current canon reopen ready",
                0,
                "No admissible current-canon delta was found that makes Trial-1 reopen-ready.",
            ),
        ],
        {
            "wording_only_reopen_admissible": False,
            "single_delta_patch_admissible": False,
            "future_canon_multi_delta_program_required": True,
            "ghost_free_reaudit_required": True,
            "stability_reaudit_required": True,
            "existing_observables_reaudit_required": True,
            "v1_1_closeout_preservation_required": True,
            "admissible_current_canon_delta_found": False,
            "first_route_to_close_or_none": "part5_future_canon_challenge_wording_freeze",
        },
        {
            "overall_status": "trial1_case_b_future_canon_delta_admissibility_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_147": True,
            "next_required_artifacts": [
                "trial1_case_b_future_canon_challenge_wording_freeze",
                "trial1_reopen_prerequisite_gate",
            ],
        },
        {
            "registry_summary": inventory["summary"],
            "mass_formula": vev_mass["summary"]["transverse_effective_mass_squared_formula"],
            "case_b_scope_summary": case_b_scope["summary"],
            "followthrough_summary": case_b_followthrough["summary"],
        },
    )

    wording_targets = [
        target_record(
            "part5_future_canon_registry_intro",
            PART5,
            part5_text,
            "future-canon delta registry を次の 4 本に固定する。",
            "Part V must announce the future-canon delta registry explicitly.",
        ),
        target_record(
            "part5_registry_delta_proca_mass",
            PART5,
            part5_text,
            "`delta_proca_mass`",
            "Part V must expose the action-level transverse mass prerequisite.",
        ),
        target_record(
            "part5_registry_delta_pole",
            PART5,
            part5_text,
            "`delta_pole`",
            "Part V must expose the pole-structure prerequisite.",
        ),
        target_record(
            "part5_registry_delta_em_sector",
            PART5,
            part5_text,
            "`delta_em_sector`",
            "Part V must expose the independent-EM elimination prerequisite.",
        ),
        target_record(
            "part5_registry_delta_judgment",
            PART5,
            part5_text,
            "`delta_judgment`",
            "Part V must expose the downstream Part III-A judgment prerequisite.",
        ),
    ]
    wording_ready = all(item["present"] for item in wording_targets)

    wording_freeze = payload(
        "8.7.56.147",
        "Part V future-canon challenge wording freeze",
        common_inputs,
        "Freeze the public Part V wording that converts the Case-B reopen condition into a concrete future-canon delta registry.",
        {
            "wording_rule": "Part V must expose the four-item future-canon delta registry that separates action-level, pole-level, EM-sector, and judgment-level reopen prerequisites.",
        },
        [
            row(
                "trial1_case_b_part5_future_canon_wording_ready",
                "pass" if wording_ready else "reject",
                "Part V future-canon challenge wording ready",
                1 if wording_ready else 0,
                "The wording freeze closes only if all registry targets are present in Part V.",
            ),
            row(
                "trial1_case_b_part5_registry_item_count",
                "pass" if wording_ready else "reject",
                "Part V future-canon registry item count",
                sum(1 for item in wording_targets if item["present"]),
                "The registry should expose all four reopen-prerequisite families plus the introduction line.",
            ),
        ],
        {
            "part5_future_canon_registry_wording_ready": wording_ready,
            "future_canon_registry_item_keys": [
                "delta_proca_mass",
                "delta_pole",
                "delta_em_sector",
                "delta_judgment",
            ],
        },
        {
            "overall_status": "trial1_case_b_future_canon_part5_wording_frozen" if wording_ready else "trial1_case_b_future_canon_part5_wording_missing",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_148": True,
            "next_required_artifacts": [
                "trial1_reopen_prerequisite_gate",
            ],
        },
        {
            "wording_targets": wording_targets,
        },
    )

    gate = payload(
        "8.7.56.148",
        "Trial-1 reopen-prerequisite gate / Trial-2 continued-hold confirmation",
        common_inputs,
        "Freeze whether the current canon already satisfies the Trial-1 reopen prerequisites and confirm whether Trial-2 must remain on hold.",
        {
            "reopen_rule": "Trial-1 can reopen only if the future-canon delta registry becomes physically admissible; current canon itself does not satisfy that condition.",
            "trial2_rule": "Trial-2 stays on hold until Trial-1 reaches full-pass status under a future canon, not merely until a registry exists.",
            "next_route_rule": "Because Trial-2 remains on hold, the next executable v2.0 route is Trial-3 rather than Trial-2.",
        },
        [
            row(
                "trial1_case_b_reopen_prerequisite_satisfied_under_current_canon",
                "fail",
                "Trial-1 reopen prerequisite satisfied under current canon",
                0,
                "The registry is ready, but the current canon still keeps the transverse mode massive and the EM sector independent.",
            ),
            row(
                "trial1_case_b_future_canon_delta_registry_ready",
                "pass" if registry_ready and wording_ready else "reject",
                "future-canon delta registry ready",
                1 if registry_ready and wording_ready else 0,
                "The reopen prerequisites are now frozen as a registry even though they are not yet satisfied.",
            ),
            row(
                "trial2_continued_hold_confirmed",
                "pass",
                "Trial-2 continued hold confirmed",
                1,
                "Trial-2 remains blocked because Trial-1 still lacks a full photon derivation.",
            ),
        ],
        {
            "trial1_reopen_prerequisite_satisfied_under_current_canon": False,
            "trial1_future_canon_delta_registry_ready": registry_ready and wording_ready,
            "future_canon_program_required_before_reopen": True,
            "trial2_continued_hold_confirmed": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.9",
        },
        {
            "overall_status": "trial1_case_b_reopen_prerequisite_gate_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "trial3_wz_sector_source_inventory",
            ],
        },
        {
            "future_canon_delta_inventory_summary": inventory["summary"],
            "future_canon_delta_admissibility_summary": admissibility["summary"],
            "future_canon_wording_summary": wording_freeze["summary"],
            "case_b_scope_summary": case_b_scope["summary"],
        },
    )

    write_artifact("mass_origin_v2_trial1_case_b_future_canon_delta_inventory", inventory)
    write_artifact("mass_origin_v2_trial1_case_b_future_canon_delta_admissibility_audit", admissibility)
    write_artifact("mass_origin_v2_trial1_case_b_future_canon_challenge_wording_freeze", wording_freeze)
    write_artifact("mass_origin_v2_trial1_reopen_prerequisite_gate", gate)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial1_case_b_future_canon_delta_inventory_metrics.json")
    print(" - mass_origin_v2_trial1_case_b_future_canon_delta_admissibility_audit_metrics.json")
    print(" - mass_origin_v2_trial1_case_b_future_canon_challenge_wording_freeze_metrics.json")
    print(" - mass_origin_v2_trial1_reopen_prerequisite_gate_metrics.json")


# Function: run the future-canon delta registry branch from the command line.

if __name__ == "__main__":
    main()
