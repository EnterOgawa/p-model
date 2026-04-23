#!/usr/bin/env python3
"""Generate 8.7.56.403-.406 Trial-2 paper-side sync reopened artifacts.

Trial-3 now closes honestly under the coupled-localization rule on the
post-photon two-component canon. This reopens the long-deferred Trial-2
paper-side sync work. The present branch does not build the paper. Instead it:

1. freezes the reopened source inventory spanning Trial-1 breakthrough,
   Trial-2 structural pass, Trial-3 closeout, and the current paper sources,
2. audits whether the paper wording is now synchronized to the current canon,
3. formalizes the Trial-2 paper-side declaration gate, and
4. refreshes the Trial-4 disposition after the declaration-prep work closes.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

TRIAL1_DECL = OUT / "mass_origin_v2_trial1_breakthrough_declaration_gate_metrics.json"
TRIAL2_DECL = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
TRIAL3_AUDIT = OUT / "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_metrics.json"
TRIAL3_GATE = OUT / "mass_origin_v2_t3_t2_coupled_localization_closeout_declaration_gate_metrics.json"
TRIAL3_DISP = OUT / "mass_origin_v2_t3_t2_paper_sync_trial4_disp_43rd_refresh_metrics.json"

NEXT_ROUTE = "8.7.56.13"
TRIAL4_NEXT_STATE = "next_official_exploratory_branch_after_trial2_paper_sync"

LEGACY_INDEPENDENT_MAXWELL = (
    "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、"
    "P-model の枠組みとは独立に採用する"
)
LEGACY_TRIAL2_HOLD = "Trial-2 は hold を継続する。"
LEGACY_FUTURE_DELTA = "future-canon delta registry"

PART1_COUPLED_RULE = "\\kappa_{\\mathrm{coupled}}^2 = m_0^2 - \\beta_n^2"
PART3A_PHOTON = "$A_\\mu=\\delta P_\\mu^T/\\sqrt{Z_P}$"
PART3A_STRUCTURAL_PASS = "foundational / structural pass (numeric α open)"
PART3A_ORIGIN_LIMIT = "局所位相 redundancy の起源判定"
PART3A_COUPLED_REFERENCE = "coupled-localization statement fixed in Part I 2.7.0"
PART5_TRIAL2_REOPENED = "Trial-2 paper-side sync は reopened した。"
PART5_NEXT_STEP = "8.7.56.403-.406"


# 関数: UTC 現在時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 path の存在を確認する。

def req(path: Path) -> None:
    """Abort immediately when a required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 text source を読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: repo 相対 POSIX path を返す。

def rel(path: Path) -> str:
    """Return a repository-relative POSIX path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 指定した部分文字列の最初の hit 行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row payload."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 schema の payload を組み立てる。

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
    """Build the standard JSON metrics payload used across the roadmap."""
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


# 関数: JSON artifact と rows CSV を side-by-side で保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Write the metrics payload as JSON and as a rows CSV sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: wording target の present/absent を監査する。

def audit_target(
    file_key: str,
    path: Path,
    text: str,
    pattern: str,
    note: str,
    expected_present: bool = True,
) -> dict:
    """Audit whether a wording target is present or intentionally absent."""
    target_hit = hit(text, pattern)
    present = target_hit is not None
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "expected_present": expected_present,
        "present": present,
        "matched_expectation": present is expected_present,
        "note": note,
        "evidence": target_hit,
    }


# 関数: source inventory を実行する。

def build_inventory(
    common_inputs: dict,
    trial1_decl: dict,
    trial2_decl: dict,
    trial3_audit: dict,
    trial3_gate: dict,
    part1_text: str,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Freeze the reopened paper-side sync inventory."""
    inventory_targets = [
        audit_target(
            "part1_coupled_localization_rule",
            PART1,
            part1_text,
            PART1_COUPLED_RULE,
            "Part I must carry the coupled-localization rule that closed Trial-3.",
        ),
        audit_target(
            "part3a_photon_definition",
            PART3A,
            part3a_text,
            PART3A_PHOTON,
            "Part III-A must state the transverse photon identification fixed by Trial-1 breakthrough.",
        ),
        audit_target(
            "part3a_structural_pass",
            PART3A,
            part3a_text,
            PART3A_STRUCTURAL_PASS,
            "Part III-A must expose the Trial-2 foundational / structural pass state.",
        ),
        audit_target(
            "part3a_coupled_reference",
            PART3A,
            part3a_text,
            PART3A_COUPLED_REFERENCE,
            "Part III-A must keep the coupled-localization rule as reference-only and defer the primary statement to Part I.",
        ),
        audit_target(
            "part5_trial2_reopened_state",
            PART5,
            part5_text,
            PART5_TRIAL2_REOPENED,
            "Part V must record that Trial-2 paper-side sync reopened after Trial-3 closeout.",
        ),
        audit_target(
            "part5_next_official_step",
            PART5,
            part5_text,
            PART5_NEXT_STEP,
            "Part V must point to the reopened Trial-2 paper-side sync branch as the next official step.",
        ),
    ]
    inventory_ready = all(item["matched_expectation"] for item in inventory_targets)
    upstream_ready = bool(
        trial1_decl["summary"]["trial1_breakthrough_pass_under_working_action"]
        and trial2_decl["summary"]["trial2_foundational_route_confirmed"]
        and trial3_gate["summary"]["trial3_two_component_closeout_pass_under_coupled_localization"]
        and trial3_gate["summary"]["trial2_paper_side_sync_reopened"]
    )

    return payload(
        "8.7.56.403",
        "trial2_paper_side_sync_reopened_source_inventory",
        common_inputs,
        "Freeze the reopened Trial-2 paper-side sync source pack spanning the Trial-1 breakthrough, the Trial-2 structural pass, the Trial-3 coupled-localization closeout, and the synchronized paper wording surfaces.",
        {
            "inventory_rule": "inventory passes only if upstream breakthrough / structural / closeout gates are all ready and the paper source pack already exposes the synced wording surfaces",
            "paper_sync_scope": "source inventory covers Part I primary localization surface, Part III-A EM wording, and Part V checkpoint wording without running a paper build",
        },
        [
            row(
                "trial2_paper_sync_reopened_inventory_complete",
                "pass",
                "Trial-2 paper-side sync reopened inventory complete",
                1,
                "The reopened source inventory pack is frozen.",
            ),
            row(
                "trial2_paper_sync_upstream_metric_pack_ready",
                "pass" if upstream_ready else "reject",
                "upstream Trial-1 / Trial-2 / Trial-3 metric pack ready",
                1 if upstream_ready else 0,
                "The paper-side sync inventory requires the breakthrough, structural pass, and coupled-localization closeout metrics simultaneously.",
            ),
            row(
                "trial2_paper_sync_present_target_count",
                "pass" if inventory_ready else "watch",
                "present paper-side sync target count",
                sum(1 for item in inventory_targets if item["present"]),
                "All source wording targets should already be visible before the audit step runs.",
            ),
            row(
                "trial2_paper_sync_missing_target_count",
                "pass" if inventory_ready else "reject",
                "missing paper-side sync target count",
                sum(1 for item in inventory_targets if not item["present"]),
                "The reopened inventory only closes when the missing count stays zero.",
            ),
        ],
        {
            "upstream_metric_pack_ready": upstream_ready,
            "inventory_ready": inventory_ready,
            "trial1_breakthrough_pass_under_working_action": trial1_decl["summary"]["trial1_breakthrough_pass_under_working_action"],
            "trial2_pass_level": trial2_decl["summary"]["trial2_pass_level"],
            "trial3_two_component_closeout_pass_under_coupled_localization": trial3_gate["summary"]["trial3_two_component_closeout_pass_under_coupled_localization"],
            "trial2_paper_side_sync_reopened": trial3_gate["summary"]["trial2_paper_side_sync_reopened"],
            "next_required_route": "trial2_paper_side_sync_reopened_audit",
        },
        {
            "overall_status": "trial2_paper_side_sync_reopened_source_inventory_frozen",
            "advance_to_8_7_56_404": True,
            "next_required_artifacts": ["trial2_paper_side_sync_reopened_audit"],
        },
        {
            "inventory_targets": inventory_targets,
            "trial1_declaration_summary": trial1_decl["summary"],
            "trial2_declaration_summary": trial2_decl["summary"],
            "trial3_audit_summary": trial3_audit["summary"],
            "trial3_gate_summary": trial3_gate["summary"],
        },
    )


# 関数: paper-side sync audit を実行する。

def build_audit(
    common_inputs: dict,
    inventory: dict,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Audit whether the paper wording is synchronized to the current canon."""
    audit_targets = [
        audit_target(
            "part3a_legacy_independent_maxwell_absent",
            PART3A,
            part3a_text,
            LEGACY_INDEPENDENT_MAXWELL,
            "The old independent-Maxwell adoption sentence must no longer remain on the public Part III-A surface.",
            expected_present=False,
        ),
        audit_target(
            "part3a_origin_judgment_limited",
            PART3A,
            part3a_text,
            PART3A_ORIGIN_LIMIT,
            "The legacy A/B judgment must now be explicitly limited to the origin question rather than used as a Trial-2 hold sentence.",
        ),
        audit_target(
            "part3a_legacy_trial2_hold_absent",
            PART3A,
            part3a_text,
            LEGACY_TRIAL2_HOLD,
            "Part III-A must no longer say that Trial-2 remains on hold.",
            expected_present=False,
        ),
        audit_target(
            "part5_legacy_future_delta_absent",
            PART5,
            part5_text,
            LEGACY_FUTURE_DELTA,
            "Part V must no longer expose the superseded future-canon delta registry text.",
            expected_present=False,
        ),
        audit_target(
            "part5_trial2_reopened_present",
            PART5,
            part5_text,
            PART5_TRIAL2_REOPENED,
            "Part V must record the reopened Trial-2 paper-side sync state.",
        ),
        audit_target(
            "part5_current_next_step_present",
            PART5,
            part5_text,
            PART5_NEXT_STEP,
            "Part V must expose the current next official step after Trial-3 closeout.",
        ),
    ]
    audit_ready = bool(inventory["summary"]["inventory_ready"]) and all(
        item["matched_expectation"] for item in audit_targets
    )

    return payload(
        "8.7.56.404",
        "trial2_paper_side_sync_reopened_audit",
        common_inputs,
        "Audit whether the paper wording now reflects the breakthrough structural EM route, the Trial-3 coupled-localization closeout, and the reopened Trial-2 ordering without preserving the old hold-era language.",
        {
            "sync_rule": "paper-side sync passes only if the stale hold-era wording is absent and the current breakthrough / structural / closeout wording is explicit on the intended surfaces",
            "ordering_rule": "Part III-A carries the EM status wording, Part I remains the primary coupled-localization surface, and Part V carries the v2 checkpoint summary",
        },
        [
            row(
                "trial2_paper_sync_reopened_audit_complete",
                "pass",
                "Trial-2 paper-side sync reopened audit complete",
                1,
                "The reopened paper-side sync audit is frozen.",
            ),
            row(
                "trial2_paper_sync_reopened_audit_ready",
                "pass" if audit_ready else "reject",
                "reopened paper-side sync audit ready",
                1 if audit_ready else 0,
                "The audit closes only when stale wording is absent and current wording is explicit.",
            ),
            row(
                "trial2_paper_sync_stale_target_count",
                "pass" if audit_ready else "watch",
                "stale wording hit count",
                sum(1 for item in audit_targets if (not item["expected_present"]) and item["present"]),
                "The stale hit count must stay zero.",
            ),
            row(
                "trial2_paper_sync_current_target_count",
                "pass" if audit_ready else "watch",
                "current wording hit count",
                sum(1 for item in audit_targets if item["expected_present"] and item["present"]),
                "The current wording hit count should include every synced surface.",
            ),
        ],
        {
            "audit_ready": audit_ready,
            "part3a_stale_independent_maxwell_removed": not audit_targets[0]["present"],
            "part3a_origin_judgment_limited_to_origin_question": audit_targets[1]["present"],
            "part3a_trial2_hold_removed": not audit_targets[2]["present"],
            "part5_future_canon_delta_removed": not audit_targets[3]["present"],
            "part5_reopened_trial2_state_present": audit_targets[4]["present"],
            "part5_current_next_step_present": audit_targets[5]["present"],
            "next_required_route": "trial2_paper_side_sync_reopened_declaration_gate",
        },
        {
            "overall_status": "trial2_paper_side_sync_reopened_audited",
            "advance_to_8_7_56_405": True,
            "next_required_artifacts": ["trial2_paper_side_sync_reopened_declaration_gate"],
        },
        {
            "inventory_summary": inventory["summary"],
            "audit_targets": audit_targets,
        },
    )


# 関数: Trial-2 declaration gate を実行する。

def build_gate(common_inputs: dict, inventory: dict, audit: dict) -> dict:
    """Freeze the reopened Trial-2 paper-side sync declaration gate."""
    gate_ready = bool(inventory["summary"]["inventory_ready"] and audit["summary"]["audit_ready"])
    return payload(
        "8.7.56.405",
        "trial2_paper_side_sync_reopened_declaration_gate",
        common_inputs,
        "Freeze the declaration after the reopened Trial-2 paper-side sync: the current paper wording is synchronized to the breakthrough / structural / closeout state, and the next mainline may move beyond the deferred Trial-2 reserve status.",
        {
            "gate_rule": "close the reopened Trial-2 paper-side sync once the current paper wording reflects Trial-1 breakthrough, Trial-2 structural pass, and Trial-3 coupled-localization closeout without stale hold-era language",
            "handoff_rule": "the paper-source sync itself closes here; paper build remains user-triggered",
        },
        [
            row(
                "trial2_paper_side_sync_reopened_gate_complete",
                "pass",
                "Trial-2 paper-side sync reopened gate complete",
                1,
                "The reopened declaration gate is frozen.",
            ),
            row(
                "trial2_paper_side_sync_reopened_complete",
                "pass" if gate_ready else "reject",
                "Trial-2 paper-side sync reopened complete",
                1 if gate_ready else 0,
                "The paper-side sync branch closes only after both the inventory and audit pass.",
            ),
            row(
                "trial2_current_paper_state_synced",
                "pass" if gate_ready else "reject",
                "current Trial-2 paper state synced",
                1 if gate_ready else 0,
                "Part I / Part III-A / Part V now carry the current canon wording for the reopened EM branch.",
            ),
            row(
                "trial2_user_build_handoff_ready",
                "pass" if gate_ready else "watch",
                "user build handoff ready",
                1 if gate_ready else 0,
                "The source-side sync is complete even though paper builds remain user-triggered.",
            ),
        ],
        {
            "trial2_paper_side_sync_reopened_complete": gate_ready,
            "trial2_current_paper_state_synced": gate_ready,
            "paper_build_user_handoff_ready": gate_ready,
            "trial4_declaration_prep_complete": gate_ready,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_paper_side_sync_reopened_complete",
            "advance_to_8_7_56_406": True,
            "next_required_artifacts": ["trial2_paper_sync_trial4_disposition_44th_refresh"],
        },
        {
            "inventory_summary": inventory["summary"],
            "audit_summary": audit["summary"],
        },
    )


# 関数: Trial-4 disposition refresh を実行する。

def build_disposition(common_inputs: dict, gate: dict, trial3_disp: dict) -> dict:
    """Refresh the Trial-4 ordering after the reopened Trial-2 sync closes."""
    trial4_released = bool(gate["summary"]["trial2_paper_side_sync_reopened_complete"])
    return payload(
        "8.7.56.406",
        "trial2_paper_sync_trial4_disposition_44th_refresh",
        common_inputs,
        "Refresh the post-sync ordering: once Trial-2 paper-side sync and declaration prep are complete, release the long-deferred Trial-4 exploratory branch as the next official route.",
        {
            "trial4_release_rule": "release Trial-4 only after the reopened Trial-2 paper-side sync and declaration prep are complete",
            "next_route_rule": "once released, the next official branch returns to Trial-4 exploratory inventory 8.7.56.13",
        },
        [
            row(
                "trial2_paper_sync_trial4_disp_44th_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition forty-fourth refresh complete",
                1,
                "The post-sync disposition refresh is frozen.",
            ),
            row(
                "trial2_paper_side_sync_completed_after_trial3_closeout",
                "pass" if trial4_released else "reject",
                "Trial-2 paper-side sync completed after Trial-3 closeout",
                1 if trial4_released else 0,
                "The reserve branch is no longer merely reopened; it is now completed.",
            ),
            row(
                "trial4_reopened_as_next_official_branch",
                "pass" if trial4_released else "reject",
                "Trial-4 reopened as next official branch",
                1 if trial4_released else 0,
                "Trial-4 may advance once the paper-side declaration prep is no longer pending.",
            ),
        ],
        {
            "trial2_paper_side_sync_state": "completed_after_trial3_closeout" if trial4_released else "reopened_after_trial3_closeout",
            "trial4_deferred": not trial4_released,
            "trial4_branch_state": TRIAL4_NEXT_STATE if trial4_released else "still_deferred",
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_paper_sync_trial4_disposition_44th_refresh_frozen",
            "advance_to_8_7_56_13": trial4_released,
            "next_required_artifacts": ["trial4_nonabelian_color_like_internal_degree_inventory"],
        },
        {
            "trial2_gate_summary": gate["summary"],
            "prior_disposition_summary": trial3_disp["summary"],
        },
    )


# 関数: current branch 全体を実行する。

def main() -> None:
    """Execute the reopened Trial-2 paper-side sync branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART1,
        PART3A,
        PART5,
        TRIAL1_DECL,
        TRIAL2_DECL,
        TRIAL3_AUDIT,
        TRIAL3_GATE,
        TRIAL3_DISP,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    trial1_decl = read_json(TRIAL1_DECL)
    trial2_decl = read_json(TRIAL2_DECL)
    trial3_audit = read_json(TRIAL3_AUDIT)
    trial3_gate = read_json(TRIAL3_GATE)
    trial3_disp = read_json(TRIAL3_DISP)

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_v2_trial1_breakthrough_declaration_gate_json": rel(TRIAL1_DECL),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECL),
        "mass_origin_v2_t3_t2_coupled_localization_closeout_audit_json": rel(TRIAL3_AUDIT),
        "mass_origin_v2_t3_t2_coupled_localization_closeout_declaration_gate_json": rel(TRIAL3_GATE),
        "mass_origin_v2_t3_t2_paper_sync_trial4_disp_43rd_refresh_json": rel(TRIAL3_DISP),
    }

    inventory = build_inventory(
        common_inputs,
        trial1_decl,
        trial2_decl,
        trial3_audit,
        trial3_gate,
        part1_text,
        part3a_text,
        part5_text,
    )
    audit = build_audit(common_inputs, inventory, part3a_text, part5_text)
    gate = build_gate(common_inputs, inventory, audit)
    disposition = build_disposition(common_inputs, gate, trial3_disp)

    write_artifact("mass_origin_v2_trial2_paper_side_sync_reopened_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_paper_side_sync_reopened_audit", audit)
    write_artifact("mass_origin_v2_trial2_paper_side_sync_reopened_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_paper_sync_trial4_disposition_44th_refresh", disposition)

    print("[done] 8.7.56.403-.406 artifacts written:")
    print(" - mass_origin_v2_trial2_paper_side_sync_reopened_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_paper_side_sync_reopened_audit_metrics.json")
    print(" - mass_origin_v2_trial2_paper_side_sync_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial2_paper_sync_trial4_disposition_44th_refresh_metrics.json")
    print(f"[context] status lines: {len(status_text.splitlines())}, roadmap lines: {len(roadmap_text.splitlines())}")
    print(f"[context] ai current step: {ai_context['current_step']}")


if __name__ == "__main__":
    main()
