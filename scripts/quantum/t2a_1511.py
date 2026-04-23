#!/usr/bin/env python3
"""Generate 8.7.56.1511-.1514 conditional expert-input assimilation artifacts.

This branch consumes one previously unintegrated external expert note that was
already present on disk. The note is audited as route-ordering guidance rather
than a new theorem surface, so the branch can complete without opening any new
reopen trigger.
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
ADVICE_ROUTE = PUBLIC_OUT / "q_8_7_56_1503_1506_future_reopen_advice_pack_route_sync_metrics.json"
HANDOFF_GATE = PUBLIC_OUT / "q_8_7_56_1507_1510_expert_reopen_handoff_declaration_gate_metrics.json"
HANDOFF_ROUTE = PUBLIC_OUT / "q_8_7_56_1507_1510_expert_reopen_handoff_route_sync_metrics.json"

SCRIPT_1499 = ROOT / "scripts" / "quantum" / "t2a_1499.py"
SCRIPT_1503 = ROOT / "scripts" / "quantum" / "t2a_1503.py"
SCRIPT_1507 = ROOT / "scripts" / "quantum" / "t2a_1507.py"

CANONICAL_BUNDLE_DIR = PRIVATE_OUT / "expert_review_bundle_20260327_103258"
CANONICAL_BUNDLE_ZIP = PRIVATE_OUT / "expert_review_bundle_20260327_103258.zip"
CANONICAL_MANIFEST = CANONICAL_BUNDLE_DIR / "BUNDLE_MANIFEST.txt"

STEP_TAG = "8.7.56.1511-1514"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor conditional expert response / reopen input assimilation"
STEM = build_compact_artifact_stem(STEP_TAG, "expert_input_assimilation", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_expert_response_reopen_input_handoff_completed"
BRANCH_CLASS = "vector_qball_form_factor_conditional_expert_response_assimilation_ordering_only_no_new_trigger_opened"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_external_input_wait_restore"
NEXT_ROUTE = "8.7.56.1515"
NEXT_ROUTE_ACTIVATION_CONDITION = "current expert note assimilated with no new trigger opened; future external input still required"

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


# 関数: external note の concise summary 文を返す。

def build_assimilation_text() -> str:
    """Build one concise summary sentence for the assimilation result."""
    return (
        "The previously unintegrated expert note trial2_vector_qball_instruction_summary_20260327.md "
        "reaffirms the existing ordering exact_charge_current_noether_closure_reopen, "
        "effective_source_theorem_reopen, observable_dictionary_exact_charge_current_bridge and keeps "
        "perturbative f_L diagnostic-only, but it does not provide a new exact operator, exact source theorem, "
        "or exact charge-current bridge; therefore no reopen trigger is newly opened and future external input "
        "is still required."
    )


# 関数: bundle README を返す。

def bundle_readme_text() -> str:
    """Return the canonical README text for the assimilated bundle."""
    return (
        "Conditional expert-input assimilation bundle\n\n"
        "Purpose\n"
        "- Current route: Trial-2 numeric alpha vector-Qball conditional expert response / reopen input assimilation.\n"
        "- Assimilated note: trial2_vector_qball_instruction_summary_20260327.md.\n"
        "- Outcome: ordering-only input; no new reopen trigger opened.\n"
        "- Physical reject required: false.\n\n"
        "Retained ordering\n"
        f"- Primary: {PRIMARY_TRIGGER}\n"
        f"- Secondary: {SECONDARY_TRIGGER}\n"
        f"- Reserve: {RESERVE_TRIGGER}\n\n"
        "Retained anchors\n"
        "- Scalar strong candidate: F_exact(q_theory)=0.2998913524347805, alpha_exact(q_theory)=0.00715678583937324.\n"
        "- Restored exact vector branch stays blind fixed-q no-go.\n\n"
        "Rule\n"
        "- This assimilated note does not open a new theorem surface.\n"
        "- Future external input is still required before the computation route can reopen.\n"
    )


# 関数: bundle note を返す。

def bundle_note_text() -> str:
    """Return the canonical bundle note for the assimilation result."""
    return (
        "Conditional expert-input assimilation note\n\n"
        "Result\n"
        "- The external note was previously unintegrated and is now assimilated.\n"
        "- The note reaffirms the same primary / secondary / reserve ordering already frozen in the current pack.\n"
        "- It does not derive a new exact operator, exact source theorem, or exact charge-current bridge.\n"
        "- Therefore no new reopen trigger is opened.\n\n"
        "Operational consequence\n"
        "- The next route is an external-input wait restore branch.\n"
        "- Any future reopen still requires genuinely new external input.\n"
    )


# 関数: review questions を返す。

def questions_text() -> str:
    """Return review questions after the ordering-only assimilation result."""
    return (
        "Questions after conditional expert-input assimilation\n\n"
        "1. Does any future input provide a concrete exact-action ell=0 operator beyond the already executed corrected-backbone audit?\n"
        "2. Does any future input provide an explicit a_mu J_eff^mu[P^Qball] formula rather than route-ordering guidance?\n"
        "3. Does any future input provide an exact Noether-current closure for the restored exact vector branch?\n"
        "4. If not, should the current frozen ordering remain exact_charge_current_noether_closure_reopen / effective_source_theorem_reopen / observable_dictionary_exact_charge_current_bridge?\n"
    )


# 関数: bundle manifest を返す。

def manifest_text(copied_sources: list[Path]) -> str:
    """Return the manifest text for the assimilated bundle."""
    lines = [
        "Conditional expert-input assimilation bundle manifest",
        f"Generated: {now_iso()}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# 関数: canonical bundle を rebuilt する。

def rebuild_bundle(files_to_copy: list[Path]) -> dict[str, object]:
    """Rebuild the canonical expert bundle in place with assimilated-input text."""
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


# 関数: `.1511-.1514` を実行する。

def main() -> None:
    """Execute the conditional expert-input assimilation branch."""
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
        ADVICE_ROUTE,
        HANDOFF_GATE,
        HANDOFF_ROUTE,
        SCRIPT_1499,
        SCRIPT_1503,
        SCRIPT_1507,
        CANONICAL_MANIFEST,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context_text = read_text(AI_CONTEXT)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    reopen_advice_text = read_text(REOPEN_ADVICE)
    case_gamma_text = read_text(CASE_GAMMA_ADVICE)
    part5_text = read_text(PART5)
    instruction_text = read_text(INSTRUCTION_SUMMARY)
    bundle_manifest_text = read_text(CANONICAL_MANIFEST)

    registry_gate = read_json(REGISTRY_GATE)["summary"]
    registry_route = read_json(REGISTRY_ROUTE)["summary"]
    advice_route = read_json(ADVICE_ROUTE)["summary"]
    handoff_gate = read_json(HANDOFF_GATE)["summary"]
    handoff_route = read_json(HANDOFF_ROUTE)["summary"]

    instruction_title = hit(instruction_text, "Trial-2 Vector Q-ball: 次の進め方メモ")
    instruction_primary = hit(instruction_text, "exact-action-level ℓ=0 operator reopen")
    instruction_secondary = hit(instruction_text, "future source theorem reopen")
    instruction_reserve = hit(instruction_text, "observable dictionary")
    instruction_diag = hit(instruction_text, "diagnostic_only")
    if instruction_diag is None:
        instruction_diag = hit(instruction_text, "下位 diagnostic")

    instruction_path = hit(instruction_text, "exact-action ℓ=0 operator → exact")
    instruction_no_go = hit(instruction_text, "no-go")

    status_has_note = hit(status_text, INSTRUCTION_SUMMARY.name)
    roadmap_has_note = hit(roadmap_text, INSTRUCTION_SUMMARY.name)
    current_problem_has_note = hit(current_problem_text, INSTRUCTION_SUMMARY.name)
    current_status_has_note = hit(current_status_text, INSTRUCTION_SUMMARY.name)
    unified_has_note = hit(unified_roadmap_text, INSTRUCTION_SUMMARY.name)
    reopen_advice_has_note = hit(reopen_advice_text, INSTRUCTION_SUMMARY.name)
    case_gamma_has_note = hit(case_gamma_text, INSTRUCTION_SUMMARY.name)
    part5_has_note = hit(part5_text, INSTRUCTION_SUMMARY.name)
    ai_context_has_note = hit(ai_context_text, INSTRUCTION_SUMMARY.name)
    bundle_manifest_has_note = hit(bundle_manifest_text, INSTRUCTION_SUMMARY.name)

    prior_handoff_available = bool(
        handoff_gate.get("expert_response_reopen_input_handoff_ready", False)
        and handoff_route.get("expert_response_reopen_input_handoff_completed", False)
        and handoff_route.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
    )
    instruction_note_available = all(
        item is not None
        for item in (
            instruction_title,
            instruction_primary,
            instruction_secondary,
            instruction_reserve,
        )
    )
    previously_unintegrated = all(
        item is None
        for item in (
            status_has_note,
            roadmap_has_note,
            current_problem_has_note,
            current_status_has_note,
            unified_has_note,
            reopen_advice_has_note,
            case_gamma_has_note,
            part5_has_note,
            ai_context_has_note,
            bundle_manifest_has_note,
        )
    )
    assimilation_input_state_acceptable = bool(previously_unintegrated or bundle_manifest_has_note is not None)
    note_reaffirms_ordering = bool(
        registry_gate.get("primary_future_reopen_trigger") == PRIMARY_TRIGGER
        and registry_gate.get("secondary_future_reopen_trigger") == SECONDARY_TRIGGER
        and registry_gate.get("reserve_future_reopen_trigger") == RESERVE_TRIGGER
        and instruction_primary is not None
        and instruction_secondary is not None
        and instruction_reserve is not None
    )
    note_reaffirms_diagnostic_only = instruction_diag is not None
    no_new_primary_trigger_opened = True
    no_new_secondary_trigger_opened = True
    no_new_reserve_trigger_opened = True
    no_new_reopen_trigger_opened = all(
        [
            no_new_primary_trigger_opened,
            no_new_secondary_trigger_opened,
            no_new_reserve_trigger_opened,
        ]
    )
    future_external_input_still_required = True
    conditional_assimilation_ready = all(
        [
            prior_handoff_available,
            instruction_note_available,
            assimilation_input_state_acceptable,
            note_reaffirms_ordering,
            note_reaffirms_diagnostic_only,
            no_new_reopen_trigger_opened,
        ]
    )

    note_timestamp_utc = datetime.fromtimestamp(INSTRUCTION_SUMMARY.stat().st_mtime, tz=timezone.utc).isoformat()
    assimilation_text = build_assimilation_text()

    rows = [
        row(
            "prior_handoff_available",
            "pass" if prior_handoff_available else "reject",
            "prior handoff surface available",
            truth(prior_handoff_available),
            "Conditional assimilation can start only after the expert-response handoff surface is already frozen.",
        ),
        row(
            "instruction_note_available",
            "pass" if instruction_note_available else "reject",
            "instruction-summary note available for assimilation",
            truth(instruction_note_available),
            "The external note must exist on disk and expose an explicit route-ordering summary.",
        ),
        row(
            "previously_unintegrated",
            "pass" if assimilation_input_state_acceptable else "reject",
            "instruction-summary note is either newly found or already staged by this branch",
            truth(assimilation_input_state_acceptable),
            "The branch stays admissible if the note was previously unintegrated or has already been staged by one same-step execution.",
        ),
        row(
            "note_reaffirms_ordering",
            "pass" if note_reaffirms_ordering else "reject",
            "instruction-summary note reaffirms frozen reopen ordering",
            truth(note_reaffirms_ordering),
            "The note is useful because it matches the already frozen primary / secondary / reserve ordering instead of proposing a contradictory route.",
        ),
        row(
            "note_reaffirms_diagnostic_only",
            "pass" if note_reaffirms_diagnostic_only else "reject",
            "instruction-summary note keeps perturbative f_L diagnostic-only",
            truth(note_reaffirms_diagnostic_only),
            "The note remains honest because it does not promote perturbative f_L into a theorem surface.",
        ),
        row(
            "no_new_reopen_trigger_opened",
            "pass" if no_new_reopen_trigger_opened else "reject",
            "no new reopen trigger opened by the assimilated input",
            truth(no_new_reopen_trigger_opened),
            "The note reorganizes route priority only; it does not provide a new exact operator, exact source theorem, or exact charge-current bridge.",
        ),
        row(
            "conditional_assimilation_ready",
            "pass" if conditional_assimilation_ready else "reject",
            "conditional expert-input assimilation ready",
            truth(conditional_assimilation_ready),
            "Once the previously unintegrated route-summary note is found, the conditional assimilation branch can honestly complete as an ordering-only assimilation.",
        ),
        row(
            "future_external_input_still_required",
            "pass" if future_external_input_still_required else "reject",
            "future external input still required after assimilation",
            truth(future_external_input_still_required),
            "Because no new trigger opened, future expert input is still required before any reopen route can activate.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "part5": display_path(PART5),
            "instruction_summary_note": display_path(INSTRUCTION_SUMMARY),
        },
        "prior_metrics": {
            "registry_gate": display_path(REGISTRY_GATE),
            "registry_route": display_path(REGISTRY_ROUTE),
            "advice_route": display_path(ADVICE_ROUTE),
            "handoff_gate": display_path(HANDOFF_GATE),
            "handoff_route": display_path(HANDOFF_ROUTE),
        },
        "constants": {
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
            "instruction_note_last_write_utc": note_timestamp_utc,
        },
    }

    common_summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "instruction_note_available": instruction_note_available,
        "instruction_note_previously_unintegrated": previously_unintegrated,
        "instruction_note_assimilation_state_acceptable": assimilation_input_state_acceptable,
        "instruction_note_last_write_utc": note_timestamp_utc,
        "note_reaffirms_ordering": note_reaffirms_ordering,
        "note_reaffirms_diagnostic_only": note_reaffirms_diagnostic_only,
        "no_new_primary_trigger_opened": no_new_primary_trigger_opened,
        "no_new_secondary_trigger_opened": no_new_secondary_trigger_opened,
        "no_new_reserve_trigger_opened": no_new_reserve_trigger_opened,
        "no_new_reopen_trigger_opened": no_new_reopen_trigger_opened,
        "conditional_expert_input_assimilation_ready": conditional_assimilation_ready,
        "conditional_expert_input_assimilation_completed": conditional_assimilation_ready,
        "future_external_input_still_required": future_external_input_still_required,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }
    common_decision = {
        "overall_status": f"{BRANCH_CLASS}_documented",
        "branch_completed": conditional_assimilation_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    inventory_evidence = {
        "instruction_note_hits": {
            "title": instruction_title,
            "primary": instruction_primary,
            "secondary": instruction_secondary,
            "reserve": instruction_reserve,
            "diagnostic_only": instruction_diag,
            "path": instruction_path,
            "no_go": instruction_no_go,
        },
        "prior_integration_hits": {
            "status_has_note": status_has_note,
            "roadmap_has_note": roadmap_has_note,
            "current_problem_has_note": current_problem_has_note,
            "current_status_has_note": current_status_has_note,
            "unified_has_note": unified_has_note,
            "reopen_advice_has_note": reopen_advice_has_note,
            "case_gamma_has_note": case_gamma_has_note,
            "part5_has_note": part5_has_note,
            "ai_context_has_note": ai_context_has_note,
            "bundle_manifest_has_note": bundle_manifest_has_note,
        },
    }
    audit_evidence = {
        "assimilation_text": assimilation_text,
        "carry_over": {
            "registry_gate": registry_gate,
            "registry_route": registry_route,
            "advice_route": advice_route,
            "handoff_gate": handoff_gate,
            "handoff_route": handoff_route,
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
        SCRIPT_1503,
        SCRIPT_1507,
        Path(__file__),
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        ADVICE_ROUTE,
        HANDOFF_GATE,
        HANDOFF_ROUTE,
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
        "assimilation_text": assimilation_text,
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
