#!/usr/bin/env python3
"""Generate 8.7.56.1523-.1526 future genuine external-input wait-hold artifacts.

This branch formalizes the explicit hold state reached after the current
Downloads candidate pool was exhausted. The route should remain dormant until a
genuinely new external expert response or reopen input arrives outside the
already integrated candidate set.
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

PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1519_1522_genuine_external_input_assimilation_declaration_gate_metrics.json"
)
PRIOR_ROUTE = (
    PUBLIC_OUT
    / "q_8_7_56_1519_1522_genuine_external_input_assimilation_route_sync_metrics.json"
)
WAIT_ROUTE = PUBLIC_OUT / "q_8_7_56_1515_1518_external_input_wait_restore_route_sync_metrics.json"

SCRIPT_1499 = ROOT / "scripts" / "quantum" / "t2a_1499.py"
SCRIPT_1507 = ROOT / "scripts" / "quantum" / "t2a_1507.py"
SCRIPT_1511 = ROOT / "scripts" / "quantum" / "t2a_1511.py"
SCRIPT_1515 = ROOT / "scripts" / "quantum" / "t2a_1515.py"
SCRIPT_1519 = ROOT / "scripts" / "quantum" / "t2a_1519.py"

CANONICAL_BUNDLE_DIR = PRIVATE_OUT / "expert_review_bundle_20260327_103258"
CANONICAL_BUNDLE_ZIP = PRIVATE_OUT / "expert_review_bundle_20260327_103258.zip"

STEP_TAG = "8.7.56.1523-1526"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor future genuine external-input wait hold"
STEM = build_compact_artifact_stem(STEP_TAG, "future_external_input_wait_hold", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_conditional_genuine_external_input_assimilation_completed_no_new_input_detected"
)
BRANCH_CLASS = "vector_qball_form_factor_future_genuine_external_input_wait_hold_completed"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_future_genuine_external_input_assimilation"
)
NEXT_ROUTE = "8.7.56.1527"
NEXT_ROUTE_ACTIVATION_CONDITION = (
    "future genuinely new external expert response or reopen input outside the exhausted current candidate pool arrives"
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


# 関数: ISO UTC 文字列を datetime へ変換する。

def parse_utc(text: str) -> datetime:
    """Parse one ISO-8601 UTC string."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


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


# 関数: live candidate note 一覧を返す。

def live_candidate_notes() -> list[Path]:
    """Return the current Downloads candidate pool for trial2 vector-Qball notes."""
    notes = [
        path
        for path in DOWNLOADS.glob("*.md")
        if any(token in path.name for token in ("trial2", "vector_qball", "pmodel_v2_trial2"))
    ]
    return sorted(notes, key=lambda item: item.name.lower())


# 関数: wait-hold summary 文を返す。

def build_hold_text() -> str:
    """Build one concise hold summary sentence."""
    return (
        "The already exhausted current candidate pool still contains no genuinely new external expert response or reopen input. "
        "The route therefore remains frozen in a future-input-only hold state, and no internal wording or computation restart "
        "is admissible until future external input arrives outside the exhausted pool."
    )


# 関数: bundle README を返す。

def bundle_readme_text() -> str:
    """Return the canonical README text for the future-input-only hold bundle."""
    return (
        "Future genuine external-input wait-hold bundle\n\n"
        "Purpose\n"
        "- Current route: Trial-2 numeric alpha vector-Qball future genuine external-input wait hold.\n"
        "- Outcome: the exhausted current candidate pool still contains no genuinely new external input.\n"
        "- Physical reject required: false.\n\n"
        "Frozen ordering\n"
        f"- Primary: {PRIMARY_TRIGGER}\n"
        f"- Secondary: {SECONDARY_TRIGGER}\n"
        f"- Reserve: {RESERVE_TRIGGER}\n\n"
        "Rule\n"
        "- Do not restart internal wording or reopen computation from the current pack.\n"
        "- Only future genuinely new external input outside the exhausted current candidate pool can activate the next branch.\n"
    )


# 関数: bundle note を返す。

def bundle_note_text() -> str:
    """Return the canonical bundle note for the future-input-only hold result."""
    return (
        "Future genuine external-input wait-hold note\n\n"
        "Result\n"
        "- The current live Downloads candidate scan matches the already exhausted candidate pool.\n"
        "- No extra candidate name was found.\n"
        "- No candidate note was updated after the prior genuine-input assimilation audit.\n"
        "- Therefore no new reopen trigger is open now.\n\n"
        "Operational consequence\n"
        "- The route remains dormant.\n"
        "- Reopen work can restart only after future genuinely new external input arrives.\n"
    )


# 関数: review questions を返す。

def questions_text() -> str:
    """Return review questions for the future-input-only hold bundle."""
    return (
        "Questions during future genuine external-input wait hold\n\n"
        "1. Does future input provide an exact charge-current / Noether-current closure for the restored exact vector branch?\n"
        "2. If not, does it provide a stronger effective source theorem than the current pack?\n"
        "3. If neither opens, should the frozen primary / secondary / reserve reopen ordering remain unchanged?\n"
    )


# 関数: manifest を返す。

def manifest_text(copied_sources: list[Path]) -> str:
    """Return the manifest text for the hold bundle."""
    lines = [
        "Future genuine external-input wait-hold bundle manifest",
        f"Generated: {now_iso()}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# 関数: canonical bundle を rebuilt する。

def rebuild_bundle(files_to_copy: list[Path]) -> dict[str, object]:
    """Rebuild the canonical expert bundle in place with future-hold text."""
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


# 関数: `.1523-.1526` を実行する。

def main() -> None:
    """Execute the future genuine external-input wait-hold branch."""
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
        PRIOR_GATE,
        PRIOR_ROUTE,
        WAIT_ROUTE,
        SCRIPT_1499,
        SCRIPT_1507,
        SCRIPT_1511,
        SCRIPT_1515,
        SCRIPT_1519,
    ):
        require(path)

    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)
    prior_wait_route = read_json(WAIT_ROUTE)

    prior_summary = prior_gate["summary"]
    prior_route_summary = prior_route["summary"]
    prior_assimilation_utc = parse_utc(prior_gate["generated_utc"])
    expected_pool = sorted(Path(path).name for path in prior_gate["inputs"]["candidate_pool"])
    live_notes = live_candidate_notes()
    live_pool = sorted(path.name for path in live_notes)

    extra_names = sorted(set(live_pool) - set(expected_pool))
    missing_names = sorted(set(expected_pool) - set(live_pool))

    candidate_rows: list[dict] = []
    updated_count = 0
    for note in live_notes:
        mtime_utc = datetime.fromtimestamp(note.stat().st_mtime, tz=timezone.utc)
        updated_since_prior_assimilation = mtime_utc > prior_assimilation_utc
        updated_count += 1 if updated_since_prior_assimilation else 0
        candidate_rows.append(
            {
                "note": display_path(note),
                "note_name": note.name,
                "mtime_utc": mtime_utc.isoformat(),
                "in_expected_exhausted_pool": note.name in expected_pool,
                "updated_since_prior_assimilation": updated_since_prior_assimilation,
            }
        )

    prior_assimilation_available = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("candidate_pool_exhausted", False)
        and prior_summary.get("no_new_input_detected", False)
    )
    prior_wait_restore_available = bool(
        prior_wait_route["summary"].get("trial2_numeric_alpha_problem_classification")
        == "vector_qball_form_factor_external_input_wait_restore_completed"
        and prior_wait_route["summary"].get("future_external_input_still_required", False)
    )
    no_extra_candidate_name_detected = len(extra_names) == 0
    no_updated_candidate_detected = updated_count == 0
    live_pool_matches_exhausted_pool = no_extra_candidate_name_detected and len(missing_names) == 0
    genuinely_new_external_input_detected_now = (not no_extra_candidate_name_detected) or (not no_updated_candidate_detected)
    future_external_input_wait_hold_honest = all(
        [
            prior_assimilation_available,
            prior_wait_restore_available,
            no_extra_candidate_name_detected,
            no_updated_candidate_detected,
            not genuinely_new_external_input_detected_now,
        ]
    )
    future_external_input_wait_hold_completed = future_external_input_wait_hold_honest

    rows = [
        row(
            "prior_assimilation_available",
            "pass" if prior_assimilation_available else "reject",
            "prior no-new-input assimilation available",
            truth(prior_assimilation_available),
            "The wait-hold branch is only honest after the current candidate pool was already proven exhausted.",
        ),
        row(
            "prior_wait_restore_available",
            "pass" if prior_wait_restore_available else "reject",
            "prior wait-restore state available",
            truth(prior_wait_restore_available),
            "The future-input-only hold should inherit the earlier explicit dormant state.",
        ),
        row(
            "expected_candidate_count",
            "pass",
            "expected exhausted-pool candidate count",
            float(len(expected_pool)),
            "This is the already integrated candidate set frozen by `.1519-.1522`.",
        ),
        row(
            "live_candidate_count",
            "pass",
            "current live candidate count",
            float(len(live_pool)),
            "This is the current Downloads trial2/vector-Qball candidate-note count.",
        ),
        row(
            "no_extra_candidate_name_detected",
            "pass" if no_extra_candidate_name_detected else "watch",
            "no extra candidate name detected outside exhausted pool",
            truth(no_extra_candidate_name_detected),
            "Any extra candidate name would open the next assimilation branch immediately.",
        ),
        row(
            "no_updated_candidate_detected",
            "pass" if no_updated_candidate_detected else "watch",
            "no candidate note updated after prior assimilation audit",
            truth(no_updated_candidate_detected),
            "A touched candidate after the prior audit would count as genuinely new external input now.",
        ),
        row(
            "live_pool_matches_exhausted_pool",
            "pass" if live_pool_matches_exhausted_pool else "watch",
            "live candidate pool matches exhausted pool exactly",
            truth(live_pool_matches_exhausted_pool),
            "A missing note is not fatal, but an exact match is the cleanest hold-state carry-over.",
        ),
        row(
            "genuinely_new_external_input_detected_now",
            "reject" if genuinely_new_external_input_detected_now else "pass",
            "genuinely new external input detected now",
            truth(genuinely_new_external_input_detected_now),
            "The wait-hold branch closes only while no genuinely new input exists now.",
        ),
        row(
            "future_external_input_wait_hold_honest",
            "pass" if future_external_input_wait_hold_honest else "reject",
            "future-input-only wait-hold wording honest",
            truth(future_external_input_wait_hold_honest),
            "The route should stay dormant once the current pool is exhausted and unchanged.",
        ),
        row(
            "future_external_input_wait_hold_completed",
            "pass" if future_external_input_wait_hold_completed else "reject",
            "future genuine external-input wait hold completed",
            truth(future_external_input_wait_hold_completed),
            "Completion here means the route is officially frozen until future external input arrives.",
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
            "prior_gate": display_path(PRIOR_GATE),
            "prior_route": display_path(PRIOR_ROUTE),
            "wait_route": display_path(WAIT_ROUTE),
        },
        "live_candidate_pool": [display_path(path) for path in live_notes],
        "constants": {
            "prior_assimilation_utc": prior_assimilation_utc.isoformat(),
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
        "prior_wait_restore_available": prior_wait_restore_available,
        "expected_candidate_count": len(expected_pool),
        "live_candidate_count": len(live_pool),
        "no_extra_candidate_name_detected": no_extra_candidate_name_detected,
        "no_updated_candidate_detected": no_updated_candidate_detected,
        "live_pool_matches_exhausted_pool": live_pool_matches_exhausted_pool,
        "genuinely_new_external_input_detected_now": genuinely_new_external_input_detected_now,
        "future_external_input_wait_hold_honest": future_external_input_wait_hold_honest,
        "future_external_input_wait_hold_completed": future_external_input_wait_hold_completed,
        "future_external_input_still_required": True,
        "internal_wording_loop_extension_admissible": False,
        "internal_restart_without_new_input_admissible": False,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }
    common_decision = {
        "overall_status": f"{BRANCH_CLASS}_documented",
        "branch_completed": future_external_input_wait_hold_completed,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }
    common_evidence = {
        "candidate_rows": candidate_rows,
        "extra_names": extra_names,
        "missing_names": missing_names,
        "hold_text": build_hold_text(),
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
        SCRIPT_1499,
        SCRIPT_1507,
        SCRIPT_1511,
        SCRIPT_1515,
        SCRIPT_1519,
        Path(__file__),
        PRIOR_GATE,
        PRIOR_ROUTE,
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
        "hold_text": build_hold_text(),
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
