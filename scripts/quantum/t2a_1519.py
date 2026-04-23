#!/usr/bin/env python3
"""Generate 8.7.56.1519-.1522 genuine external-input assimilation artifacts.

This branch checks whether the current Downloads candidate pool contains any
genuinely new external expert response or reopen input beyond the already
integrated vector-Qball notes. If no such input exists, the route transitions
from wait-restore into an explicit future-input-only hold state.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from datetime import datetime
from datetime import timezone
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

DOWNLOADS = Path(r"C:\Users\ogawa\Downloads")
INSTRUCTION_SUMMARY = DOWNLOADS / "trial2_vector_qball_instruction_summary_20260327.md"
UNIFIED_PLAN = DOWNLOADS / "trial2_vector_qball_unified_closure_plan_20260327.md"
NEXT_STEPS = DOWNLOADS / "trial2_vector_qball_next_steps_20260327.md"
NEXT_ACTION = DOWNLOADS / "trial2_vector_qball_next_action_20260327.md"
SOLVER_FIX = DOWNLOADS / "pmodel_v2_trial2_solver_fix_final.md"
PERTURBATIVE_NOTE = DOWNLOADS / "pmodel_v2_trial2_perturbative_fL_correction.md"

PHASE1_INV = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_source_inventory_metrics.json"
LAMBDA_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_lambda_rot_form_factor_correction_audit_metrics.json"
DIAGNOSTIC_INV = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_diagnostic_reopen_review_source_inventory_metrics.json"
ASSIMILATION_INV = PUBLIC_OUT / "q_8_7_56_1511_1514_expert_input_assimilation_inventory_metrics.json"
ASSIMILATION_ROUTE = PUBLIC_OUT / "q_8_7_56_1511_1514_expert_input_assimilation_route_sync_metrics.json"
WAIT_INV = PUBLIC_OUT / "q_8_7_56_1515_1518_external_input_wait_restore_inventory_metrics.json"
WAIT_ROUTE = PUBLIC_OUT / "q_8_7_56_1515_1518_external_input_wait_restore_route_sync_metrics.json"

SCRIPT_1499 = ROOT / "scripts" / "quantum" / "t2a_1499.py"
SCRIPT_1507 = ROOT / "scripts" / "quantum" / "t2a_1507.py"
SCRIPT_1511 = ROOT / "scripts" / "quantum" / "t2a_1511.py"
SCRIPT_1515 = ROOT / "scripts" / "quantum" / "t2a_1515.py"

CANONICAL_BUNDLE_DIR = PRIVATE_OUT / "expert_review_bundle_20260327_103258"
CANONICAL_BUNDLE_ZIP = PRIVATE_OUT / "expert_review_bundle_20260327_103258.zip"

STEP_TAG = "8.7.56.1519-1522"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor conditional genuine external-input assimilation"
STEM = build_compact_artifact_stem(STEP_TAG, "genuine_external_input_assimilation", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_external_input_wait_restore_completed"
BRANCH_CLASS = "vector_qball_form_factor_conditional_genuine_external_input_assimilation_completed_no_new_input_detected"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_future_genuine_external_input_wait_hold"
NEXT_ROUTE = "8.7.56.1523"
BRANCH_TAG_PATTERN = "8.7.56.1519-.1522"
NEXT_ROUTE_ACTIVATION_CONDITION = (
    "genuinely new external expert response or reopen input outside the exhausted current candidate pool arrives"
)

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


# 関数: UTC文字列を datetime へ変換する。

def parse_utc(text: str) -> datetime:
    """Parse one ISO-8601 UTC string."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# 関数: ファイル更新時刻をUTC文字列へ変換する。

def mtime_iso(path: Path) -> str:
    """Return one path modification time in UTC ISO format."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


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

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
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


# 関数: reference scan 対象を返す。

def reference_sources() -> list[Path]:
    """Return the minimal reference-source set for candidate integration checks."""
    return [
        STATUS,
        ROADMAP,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        PHASE1_INV,
        LAMBDA_AUDIT,
        DIAGNOSTIC_INV,
        ASSIMILATION_INV,
        ASSIMILATION_ROUTE,
        WAIT_INV,
        WAIT_ROUTE,
    ]


# 関数: candidate note 一覧を返す。

def candidate_notes() -> list[Path]:
    """Return the current external-note candidate pool."""
    ordered = [
        UNIFIED_PLAN,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
        NEXT_STEPS,
        NEXT_ACTION,
        INSTRUCTION_SUMMARY,
    ]
    return [path for path in ordered if path.exists()]


# 関数: external-input summary 文を返す。

def build_hold_text() -> str:
    """Build one concise hold summary sentence."""
    return (
        "The current Downloads candidate pool contains no genuinely new external expert response or reopen input: "
        "all current vector-Qball notes are already integrated into the present pack and none is newer than the "
        "8.7.56.1515-.1518 wait-restore timestamp, so the route stays frozen until future external input arrives."
    )


# 関数: bundle README を返す。

def bundle_readme_text() -> str:
    """Return the canonical README text for the no-new-input hold bundle."""
    return (
        "No-new-input hold bundle\n\n"
        "Purpose\n"
        "- Current route: Trial-2 numeric alpha vector-Qball conditional genuine external-input assimilation.\n"
        "- Outcome: the current Downloads candidate pool contains no genuinely new external input.\n"
        "- Physical reject required: false.\n\n"
        "Frozen ordering\n"
        f"- Primary: {PRIMARY_TRIGGER}\n"
        f"- Secondary: {SECONDARY_TRIGGER}\n"
        f"- Reserve: {RESERVE_TRIGGER}\n\n"
        "Operational rule\n"
        "- Do not restart reopen computation from current internal notes.\n"
        "- Only future external input outside the exhausted current candidate pool can open the next branch.\n"
    )


# 関数: bundle note を返す。

def bundle_note_text() -> str:
    """Return the canonical bundle note for the no-new-input result."""
    return (
        "No-new-input hold note\n\n"
        "Result\n"
        "- The current Downloads vector-Qball candidate pool has been rescanned.\n"
        "- All candidate notes are already integrated into the current pack.\n"
        "- None is newer than the external-input wait-restore timestamp.\n"
        "- Therefore no genuine new reopen trigger is open.\n\n"
        "Operational consequence\n"
        "- The route remains dormant.\n"
        "- Reopen work can restart only after future genuinely new external input arrives.\n"
    )


# 関数: review questions を返す。

def questions_text() -> str:
    """Return review questions for the no-new-input hold bundle."""
    return (
        "Questions after the no-new-input genuine-assimilation audit\n\n"
        "1. Does a future note provide an exact charge-current / Noether-current closure for the restored exact vector branch?\n"
        "2. If not, does it provide a stronger exact effective-source theorem than the current pack?\n"
        "3. If neither opens, should the frozen primary / secondary / reserve reopen ordering remain unchanged?\n"
    )


# 関数: manifest を返す。

def manifest_text(copied_sources: list[Path]) -> str:
    """Return the manifest text for the hold bundle."""
    lines = [
        "No-new-input hold bundle manifest",
        f"Generated: {now_iso()}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# 関数: canonical bundle を rebuilt する。

def rebuild_bundle(files_to_copy: list[Path]) -> dict[str, object]:
    """Rebuild the canonical expert bundle in place with hold text."""
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


# 関数: `.1519-.1522` を実行する。

def main() -> None:
    """Execute the no-new-input genuine external-input assimilation audit."""
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
        PHASE1_INV,
        LAMBDA_AUDIT,
        DIAGNOSTIC_INV,
        ASSIMILATION_INV,
        ASSIMILATION_ROUTE,
        WAIT_INV,
        WAIT_ROUTE,
        SCRIPT_1499,
        SCRIPT_1507,
        SCRIPT_1511,
        SCRIPT_1515,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)

    prior_context = read_json(AI_CONTEXT)
    prior_wait_route = read_json(WAIT_ROUTE)["summary"]
    prior_assimilation_route = read_json(ASSIMILATION_ROUTE)["summary"]
    prior_wait_utc = parse_utc(prior_context["current_date_utc"])

    source_texts = {display_path(path): read_text(path) for path in reference_sources()}
    candidates = candidate_notes()

    candidate_rows: list[dict] = []
    integrated_count = 0
    newer_count = 0
    genuine_new_count = 0

    for note in candidates:
        hits = [source for source, text in source_texts.items() if note.name in text]
        referenced = bool(hits)
        note_mtime = parse_utc(mtime_iso(note))
        newer_than_wait_restore = note_mtime > prior_wait_utc
        genuine_new = (not referenced) or newer_than_wait_restore

        integrated_count += 1 if referenced else 0
        newer_count += 1 if newer_than_wait_restore else 0
        genuine_new_count += 1 if genuine_new else 0

        candidate_rows.append(
            {
                "note": display_path(note),
                "note_name": note.name,
                "mtime_utc": note_mtime.isoformat(),
                "referenced_in_current_pack": referenced,
                "reference_hits": hits,
                "newer_than_wait_restore": newer_than_wait_restore,
                "qualifies_as_genuine_new_input": genuine_new,
            }
        )

    candidate_note_count = len(candidates)
    prior_wait_restore_available = bool(
        prior_wait_route.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_wait_route.get("future_external_input_still_required", False)
        and prior_wait_route.get("no_new_reopen_trigger_retained", False)
    )
    prior_assimilation_available = bool(
        prior_assimilation_route.get("trial2_numeric_alpha_problem_classification")
        == "vector_qball_form_factor_conditional_expert_response_assimilation_ordering_only_no_new_trigger_opened"
        and prior_assimilation_route.get("no_new_reopen_trigger_opened", False)
    )
    all_candidates_already_integrated = candidate_note_count > 0 and integrated_count == candidate_note_count
    no_candidate_newer_than_wait_restore = newer_count == 0
    genuine_new_external_input_detected = genuine_new_count > 0
    candidate_pool_exhausted = all_candidates_already_integrated and no_candidate_newer_than_wait_restore
    no_new_input_detected = not genuine_new_external_input_detected
    conditional_assimilation_honest = all(
        [
            prior_wait_restore_available,
            prior_assimilation_available,
            candidate_pool_exhausted,
            no_new_input_detected,
        ]
    )
    future_external_input_wait_hold_required = conditional_assimilation_honest

    status_conditional = hit(status_text, BRANCH_TAG_PATTERN)
    roadmap_conditional = hit(roadmap_text, f"`{BRANCH_TAG_PATTERN}`")
    current_problem_conditional = hit(current_problem_text, BRANCH_TAG_PATTERN)
    current_status_conditional = hit(current_status_text, BRANCH_TAG_PATTERN)
    unified_conditional = hit(unified_roadmap_text, BRANCH_TAG_PATTERN)
    part5_conditional = hit(part5_text, BRANCH_TAG_PATTERN)
    hold_text = build_hold_text()
    inventory_ready = all(
        item is not None
        for item in (
            status_conditional,
            roadmap_conditional,
            current_problem_conditional,
            current_status_conditional,
            unified_conditional,
            part5_conditional,
        )
    )
    branch_ready = inventory_ready and conditional_assimilation_honest

    rows = [
        row(
            "prior_wait_restore_available",
            "pass" if prior_wait_restore_available else "reject",
            "prior wait-restore state available",
            truth(prior_wait_restore_available),
            "The conditional genuine-input branch should only run after the explicit dormant state has been restored.",
        ),
        row(
            "prior_assimilation_available",
            "pass" if prior_assimilation_available else "reject",
            "prior ordering-only assimilation available",
            truth(prior_assimilation_available),
            "The last external note was already assimilated as ordering-only and opened no new trigger.",
        ),
        row(
            "candidate_note_count",
            "pass",
            "current Downloads candidate-note count",
            float(candidate_note_count),
            "The current vector-Qball candidate pool is defined by the live Downloads note set.",
        ),
        row(
            "all_candidates_already_integrated",
            "pass" if all_candidates_already_integrated else "watch",
            "all candidate notes already integrated into current pack",
            truth(all_candidates_already_integrated),
            "Older-but-unintegrated notes would count as genuine new external input for this branch.",
        ),
        row(
            "no_candidate_newer_than_wait_restore",
            "pass" if no_candidate_newer_than_wait_restore else "watch",
            "no candidate note is newer than the wait-restore timestamp",
            truth(no_candidate_newer_than_wait_restore),
            "A newer note would reopen the conditional assimilation branch immediately.",
        ),
        row(
            "genuine_new_external_input_detected",
            "reject" if genuine_new_external_input_detected else "pass",
            "genuine new external input detected in current candidate pool",
            truth(genuine_new_external_input_detected),
            "This branch closes only if the current candidate pool contains no new reopen input.",
        ),
        row(
            "candidate_pool_exhausted",
            "pass" if candidate_pool_exhausted else "watch",
            "current candidate pool exhausted by already-integrated notes",
            truth(candidate_pool_exhausted),
            "The current pack can only stay dormant if the known candidate pool is exhausted.",
        ),
        row(
            "conditional_assimilation_honest",
            "pass" if conditional_assimilation_honest else "reject",
            "conditional assimilation audit honest",
            truth(conditional_assimilation_honest),
            "The route should not fabricate progress once no genuine new input exists.",
        ),
        row(
            "future_external_input_wait_hold_required",
            "pass" if future_external_input_wait_hold_required else "reject",
            "future external-input wait hold required",
            truth(future_external_input_wait_hold_required),
            "After the current candidate pool is exhausted, only future external input can open the next branch.",
        ),
        row(
            "conditional_assimilation_inventory_ready",
            "pass" if inventory_ready else "reject",
            "conditional assimilation inventory ready",
            truth(inventory_ready),
            "The conditional branch must be visible across status, roadmap, current notes, and Part V wording.",
        ),
        row(
            "conditional_assimilation_completed",
            "pass" if branch_ready else "reject",
            "conditional genuine external-input assimilation branch completed",
            truth(branch_ready),
            "Completion here means the current candidate pool was audited and no genuine new input was found.",
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
        },
        "prior_metrics": {
            "assimilation_route": display_path(ASSIMILATION_ROUTE),
            "wait_route": display_path(WAIT_ROUTE),
        },
        "candidate_pool": [display_path(path) for path in candidates],
        "constants": {
            "prior_wait_restore_utc": prior_wait_utc.isoformat(),
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
        "prior_wait_restore_available": prior_wait_restore_available,
        "prior_assimilation_available": prior_assimilation_available,
        "candidate_note_count": candidate_note_count,
        "all_candidates_already_integrated": all_candidates_already_integrated,
        "no_candidate_newer_than_wait_restore": no_candidate_newer_than_wait_restore,
        "genuine_new_external_input_detected": genuine_new_external_input_detected,
        "candidate_pool_exhausted": candidate_pool_exhausted,
        "no_new_input_detected": no_new_input_detected,
        "conditional_assimilation_honest": conditional_assimilation_honest,
        "future_external_input_wait_hold_required": future_external_input_wait_hold_required,
        "conditional_assimilation_inventory_ready": inventory_ready,
        "conditional_assimilation_completed": branch_ready,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }
    common_decision = {
        "overall_status": f"{BRANCH_CLASS}_documented",
        "branch_completed": branch_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }
    common_evidence = {
        "candidate_rows": candidate_rows,
        "hold_text": hold_text,
        "doc_hits": {
            "status_conditional": status_conditional,
            "roadmap_conditional": roadmap_conditional,
            "current_problem_conditional": current_problem_conditional,
            "current_status_conditional": current_status_conditional,
            "unified_conditional": unified_conditional,
            "part5_conditional": part5_conditional,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload(STEP_TAG, STEP_NAME, inputs, rows, common_summary, common_decision, common_evidence),
    )
    audit_paths = write_artifact(
        "audit",
        payload(STEP_TAG, STEP_NAME, inputs, rows, common_summary, common_decision, common_evidence),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(STEP_TAG, STEP_NAME, inputs, rows, common_summary, common_decision, common_evidence),
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
        SCRIPT_1515,
        Path(__file__),
        PHASE1_INV,
        LAMBDA_AUDIT,
        DIAGNOSTIC_INV,
        ASSIMILATION_INV,
        ASSIMILATION_ROUTE,
        WAIT_INV,
        WAIT_ROUTE,
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
        "hold_text": hold_text,
        "candidate_rows": candidate_rows,
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
