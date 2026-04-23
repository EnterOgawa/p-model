#!/usr/bin/env python3
"""Generate 8.7.56.1515-.1518 external-input wait-restore artifacts.

This branch freezes the post-assimilation wait state after the previously
unintegrated instruction-summary note was consumed and no new reopen trigger was
opened. The route remains blocked on genuinely new external input.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIVATE_OUT = ROOT / "output" / "private" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
REOPEN_ADVICE = ROOT / "doc" / "quantum" / "40_trial2_numeric_alpha_vector_qball_reopen_advice_request.md"
CASE_GAMMA_ADVICE = ROOT / "doc" / "quantum" / "42_trial2_numeric_alpha_vector_qball_case_gamma_advice_request.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

INSTRUCTION_SUMMARY = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_instruction_summary_20260327.md")
UNIFIED_PLAN = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_unified_closure_plan_20260327.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")
NEXT_ACTION = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_action_20260327.md")
SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
PERTURBATIVE_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_perturbative_fL_correction.md")

REGISTRY_GATE = PUBLIC_OUT / "q_8_7_56_1499_1502_future_reopen_registry_declaration_gate_metrics.json"
REGISTRY_ROUTE = PUBLIC_OUT / "q_8_7_56_1499_1502_future_reopen_registry_route_sync_metrics.json"
HANDOFF_ROUTE = PUBLIC_OUT / "q_8_7_56_1507_1510_expert_reopen_handoff_route_sync_metrics.json"
ASSIMILATION_GATE = PUBLIC_OUT / "q_8_7_56_1511_1514_expert_input_assimilation_declaration_gate_metrics.json"
ASSIMILATION_ROUTE = PUBLIC_OUT / "q_8_7_56_1511_1514_expert_input_assimilation_route_sync_metrics.json"

SCRIPT_1499 = ROOT / "scripts" / "quantum" / "t2a_1499.py"
SCRIPT_1507 = ROOT / "scripts" / "quantum" / "t2a_1507.py"
SCRIPT_1511 = ROOT / "scripts" / "quantum" / "t2a_1511.py"

CANONICAL_BUNDLE_DIR = PRIVATE_OUT / "expert_review_bundle_20260327_103258"
CANONICAL_BUNDLE_ZIP = PRIVATE_OUT / "expert_review_bundle_20260327_103258.zip"

STEP_TAG = "8.7.56.1515-1518"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor external-input wait restore"
STEM = build_compact_artifact_stem(STEP_TAG, "external_input_wait_restore", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_conditional_expert_response_assimilation_ordering_only_no_new_trigger_opened"
BRANCH_CLASS = "vector_qball_form_factor_external_input_wait_restore_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_conditional_genuine_external_input_assimilation"
NEXT_ROUTE = "8.7.56.1519"
NEXT_ROUTE_ACTIVATION_CONDITION = "genuinely new external expert response or reopen input arrives"

PRIMARY_TRIGGER = "exact_charge_current_noether_closure_reopen"
SECONDARY_TRIGGER = "effective_source_theorem_reopen"
RESERVE_TRIGGER = "observable_dictionary_exact_charge_current_bridge"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Fail when one required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: repo 相対の表示パスへ変換する。

def display_path(path: Path) -> str:
    """Convert one absolute path into repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: compact stem で JSON / CSV を出力する。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and its rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: wait-restore summary 文を返す。

def build_wait_text() -> str:
    """Build one concise wait-restore summary sentence."""
    return (
        "After the ordering-only assimilation of trial2_vector_qball_instruction_summary_20260327.md, "
        "the current pack remains in a no-new-trigger-opened state: exact_charge_current_noether_closure_reopen, "
        "effective_source_theorem_reopen, and observable_dictionary_exact_charge_current_bridge stay frozen as "
        "primary, secondary, and reserve, physical reject remains false, and future reopen work should stay dormant "
        "until genuinely new external input arrives."
    )


# 関数: bundle README を返す。

def bundle_readme_text() -> str:
    """Return the canonical README text for the wait-restore bundle."""
    return (
        "External-input wait-restore bundle\n\n"
        "Purpose\n"
        "- Current route: Trial-2 numeric alpha vector-Qball external-input wait restore.\n"
        "- Outcome: no new reopen trigger opened after the ordering-only assimilation note.\n"
        "- Physical reject required: false.\n\n"
        "Frozen ordering\n"
        f"- Primary: {PRIMARY_TRIGGER}\n"
        f"- Secondary: {SECONDARY_TRIGGER}\n"
        f"- Reserve: {RESERVE_TRIGGER}\n\n"
        "Retained anchors\n"
        "- Scalar strong candidate: F_exact(q_theory)=0.2998913524347805, alpha_exact(q_theory)=0.00715678583937324.\n"
        "- Restored exact vector branch stays blind fixed-q no-go.\n\n"
        "Rule\n"
        "- The current pack is now in explicit wait state.\n"
        "- Reopen work should restart only when genuinely new external input arrives.\n"
    )


# 関数: bundle note を返す。

def bundle_note_text() -> str:
    """Return the canonical bundle note for the wait-restore result."""
    return (
        "External-input wait-restore note\n\n"
        "Result\n"
        "- The previously unintegrated instruction-summary note has already been assimilated.\n"
        "- It opened no new theorem surface.\n"
        "- The current route is restored to explicit wait state.\n\n"
        "Operational consequence\n"
        "- Do not reopen current-pack computation or wording loops from this state.\n"
        "- Only genuinely new external expert response or reopen input can activate the next branch.\n"
    )


# 関数: review questions を返す。

def questions_text() -> str:
    """Return review questions for the wait-restore bundle."""
    return (
        "Questions during external-input wait restore\n\n"
        "1. Does future input provide an exact charge-current / Noether-current closure for the restored exact vector branch?\n"
        "2. If not, does it provide an explicit effective source theorem stronger than the current ordering-only guidance?\n"
        "3. If neither opens, should the frozen primary / secondary / reserve ordering remain unchanged?\n"
    )


# 関数: manifest を返す。

def manifest_text(copied_sources: list[Path]) -> str:
    """Return the manifest text for the wait-restore bundle."""
    lines = [
        "External-input wait-restore bundle manifest",
        f"Generated: {now_iso()}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# 関数: canonical bundle を rebuilt する。

def rebuild_bundle(files_to_copy: list[Path]) -> dict[str, object]:
    """Rebuild the canonical expert bundle in place with wait-restore text."""
    if CANONICAL_BUNDLE_DIR.exists():
        shutil.rmtree(CANONICAL_BUNDLE_DIR)

    if CANONICAL_BUNDLE_ZIP.exists():
        CANONICAL_BUNDLE_ZIP.unlink()

    CANONICAL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    copied_sources: list[Path] = []
    for source in files_to_copy:
        require(source)
        shutil.copy2(source, CANONICAL_BUNDLE_DIR / source.name)
        copied_sources.append(source)

    (CANONICAL_BUNDLE_DIR / "README.txt").write_text(bundle_readme_text(), encoding="utf-8")
    (CANONICAL_BUNDLE_DIR / "BUNDLE_NOTE.txt").write_text(bundle_note_text(), encoding="utf-8")
    (CANONICAL_BUNDLE_DIR / "QUESTIONS_FOR_REVIEW.txt").write_text(questions_text(), encoding="utf-8")
    (CANONICAL_BUNDLE_DIR / "BUNDLE_MANIFEST.txt").write_text(manifest_text(copied_sources), encoding="utf-8")

    with zipfile.ZipFile(CANONICAL_BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(CANONICAL_BUNDLE_DIR.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)

    with zipfile.ZipFile(CANONICAL_BUNDLE_ZIP, "r") as archive:
        zip_file_count = len(archive.namelist())

    return {
        "copied_count": len(copied_sources),
        "staging_file_count": len(list(CANONICAL_BUNDLE_DIR.iterdir())),
        "zip_file_count": zip_file_count,
        "bundle_dir": display_path(CANONICAL_BUNDLE_DIR),
        "bundle_zip": display_path(CANONICAL_BUNDLE_ZIP),
    }


# 関数: `.1515-.1518` を実行する。

def main() -> None:
    """Execute the external-input wait-restore branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PRIMARY_SOURCES,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        REOPEN_ADVICE,
        CASE_GAMMA_ADVICE,
        PART1,
        PART3A,
        PART5,
        INSTRUCTION_SUMMARY,
        UNIFIED_PLAN,
        NEXT_STEPS,
        NEXT_ACTION,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        HANDOFF_ROUTE,
        ASSIMILATION_GATE,
        ASSIMILATION_ROUTE,
        SCRIPT_1499,
        SCRIPT_1507,
        SCRIPT_1511,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    registry_gate = read_json(REGISTRY_GATE)["summary"]
    registry_route = read_json(REGISTRY_ROUTE)["summary"]
    handoff_route = read_json(HANDOFF_ROUTE)["summary"]
    assimilation_gate = read_json(ASSIMILATION_GATE)["summary"]
    assimilation_route = read_json(ASSIMILATION_ROUTE)["summary"]

    status_wait = hit(status_text, "external-input wait restore")
    roadmap_wait = hit(roadmap_text, "`8.7.56.1515-.1518`")
    current_problem_wait = hit(current_problem_text, "external-input wait restore")
    current_status_wait = hit(current_status_text, "external-input wait restore")
    unified_wait = hit(unified_roadmap_text, "external-input wait restore")
    part5_wait = hit(part5_text, "external-input wait restore")

    prior_assimilation_available = bool(
        assimilation_gate.get("conditional_expert_input_assimilation_completed", False)
        and assimilation_route.get("conditional_expert_input_assimilation_completed", False)
        and assimilation_route.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
    )
    no_new_reopen_trigger_retained = bool(
        assimilation_gate.get("no_new_reopen_trigger_opened", False)
        and assimilation_route.get("no_new_reopen_trigger_opened", False)
    )
    future_external_input_still_required = bool(
        assimilation_gate.get("future_external_input_still_required", False)
        and assimilation_route.get("future_external_input_still_required", False)
    )
    frozen_ordering_retained = bool(
        registry_gate.get("primary_future_reopen_trigger") == PRIMARY_TRIGGER
        and registry_gate.get("secondary_future_reopen_trigger") == SECONDARY_TRIGGER
        and registry_gate.get("reserve_future_reopen_trigger") == RESERVE_TRIGGER
        and handoff_route.get("reopen_ordering_retained", False)
    )
    wait_restore_inventory_ready = all(
        item is not None
        for item in (
            status_wait,
            roadmap_wait,
            current_problem_wait,
            current_status_wait,
            unified_wait,
            part5_wait,
        )
    )
    wait_text = build_wait_text()
    wait_restore_honest = all(
        [
            prior_assimilation_available,
            no_new_reopen_trigger_retained,
            future_external_input_still_required,
            frozen_ordering_retained,
        ]
    )
    wait_restore_ready = all([wait_restore_inventory_ready, wait_restore_honest])

    rows = [
        row(
            "prior_assimilation_available",
            "pass" if prior_assimilation_available else "reject",
            "ordering-only assimilation available before wait restore",
            truth(prior_assimilation_available),
            "Wait restore is only honest after the ordering-only assimilation branch has actually completed.",
        ),
        row(
            "no_new_reopen_trigger_retained",
            "pass" if no_new_reopen_trigger_retained else "reject",
            "no-new-trigger state retained into wait restore",
            truth(no_new_reopen_trigger_retained),
            "The restore branch exists to freeze the explicit no-new-trigger-opened state rather than silently forgetting it.",
        ),
        row(
            "future_external_input_still_required",
            "pass" if future_external_input_still_required else "reject",
            "future external input still required",
            truth(future_external_input_still_required),
            "The route stays dormant until genuinely new external input arrives.",
        ),
        row(
            "frozen_ordering_retained",
            "pass" if frozen_ordering_retained else "reject",
            "frozen reopen ordering retained in wait restore",
            truth(frozen_ordering_retained),
            "The wait state is only meaningful if primary / secondary / reserve ordering stays unchanged.",
        ),
        row(
            "wait_restore_inventory_ready",
            "pass" if wait_restore_inventory_ready else "reject",
            "wait-restore inventory ready",
            truth(wait_restore_inventory_ready),
            "The wait route must be visible across current notes, roadmap, and Part V wording.",
        ),
        row(
            "wait_restore_honest",
            "pass" if wait_restore_honest else "reject",
            "wait-restore wording honest",
            truth(wait_restore_honest),
            "The wait state must not overclaim a reopen or hide the no-new-trigger result.",
        ),
        row(
            "external_input_wait_restore_ready",
            "pass" if wait_restore_ready else "reject",
            "external-input wait restore ready",
            truth(wait_restore_ready),
            "Once the ordering-only assimilation is complete, the next honest internal action is to restore an explicit wait state.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "part5": display_path(PART5),
            "instruction_summary_note": display_path(INSTRUCTION_SUMMARY),
        },
        "prior_metrics": {
            "registry_gate": display_path(REGISTRY_GATE),
            "registry_route": display_path(REGISTRY_ROUTE),
            "handoff_route": display_path(HANDOFF_ROUTE),
            "assimilation_gate": display_path(ASSIMILATION_GATE),
            "assimilation_route": display_path(ASSIMILATION_ROUTE),
        },
        "constants": {
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
        },
    }

    common_summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "prior_assimilation_available": prior_assimilation_available,
        "no_new_reopen_trigger_retained": no_new_reopen_trigger_retained,
        "future_external_input_still_required": future_external_input_still_required,
        "frozen_ordering_retained": frozen_ordering_retained,
        "wait_restore_inventory_ready": wait_restore_inventory_ready,
        "wait_restore_honest": wait_restore_honest,
        "external_input_wait_restore_ready": wait_restore_ready,
        "external_input_wait_restore_completed": wait_restore_ready,
        "internal_wording_loop_extension_admissible": False,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }
    common_decision = {
        "overall_status": f"{BRANCH_CLASS}_documented",
        "branch_completed": wait_restore_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    inventory_evidence = {
        "doc_hits": {
            "status_wait": status_wait,
            "roadmap_wait": roadmap_wait,
            "current_problem_wait": current_problem_wait,
            "current_status_wait": current_status_wait,
            "unified_wait": unified_wait,
            "part5_wait": part5_wait,
        }
    }
    audit_evidence = {
        "wait_text": wait_text,
        "carry_over": {
            "registry_gate": registry_gate,
            "registry_route": registry_route,
            "handoff_route": handoff_route,
            "assimilation_gate": assimilation_gate,
            "assimilation_route": assimilation_route,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload(STEP_TAG, STEP_NAME, inputs, rows, common_summary, common_decision, inventory_evidence),
    )
    audit_paths = write_artifact(
        "audit",
        payload(STEP_TAG, STEP_NAME, inputs, rows, common_summary, common_decision, audit_evidence),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(STEP_TAG, STEP_NAME, inputs, rows, common_summary, common_decision, audit_evidence),
    )

    refresh_candidates = [
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        PRIMARY_SOURCES,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        REOPEN_ADVICE,
        CASE_GAMMA_ADVICE,
        PART1,
        PART3A,
        PART5,
        INSTRUCTION_SUMMARY,
        UNIFIED_PLAN,
        NEXT_STEPS,
        NEXT_ACTION,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
        SCRIPT_1499,
        SCRIPT_1507,
        SCRIPT_1511,
        Path(__file__),
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        HANDOFF_ROUTE,
        ASSIMILATION_GATE,
        ASSIMILATION_ROUTE,
        ROOT / inventory_paths["json"],
        ROOT / audit_paths["json"],
        ROOT / gate_paths["json"],
    ]
    bundle_stats = rebuild_bundle(refresh_candidates)

    route_summary = {
        **common_summary,
        "share_pack_bundle_dir": bundle_stats["bundle_dir"],
        "share_pack_bundle_zip": bundle_stats["bundle_zip"],
        "share_pack_bundle_refresh_count": bundle_stats["copied_count"],
        "share_pack_staging_file_count": bundle_stats["staging_file_count"],
        "share_pack_zip_file_count": bundle_stats["zip_file_count"],
        "numeric_state_changed_by_current_branch": False,
        "route_state_changed_by_current_branch": True,
    }
    route_evidence = {
        "bundle_stats": bundle_stats,
        "wait_text": wait_text,
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": -0.083735013520183,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
        },
    }
    write_artifact(
        "route_sync",
        payload(STEP_TAG, STEP_NAME, inputs, rows, route_summary, common_decision, route_evidence),
    )

    print(f"[ok] wrote compact artifacts for {STEP_TAG}: {STEM}")


if __name__ == "__main__":
    main()
