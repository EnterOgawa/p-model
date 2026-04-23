#!/usr/bin/env python3
"""Generate 8.7.56.1503-.1506 future reopen advice-pack refresh artifacts.

This branch is the final internal wording step after the future reopen ordering
registry. It refreshes the expert-facing advice pack so the current state can be
handed off honestly without re-entering a wording-only loop.
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

UNIFIED_PLAN = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_unified_closure_plan_20260327.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")
NEXT_ACTION = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_action_20260327.md")
SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
PERTURBATIVE_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_perturbative_fL_correction.md")

CHARGE_CLOSURE_GATE = PUBLIC_OUT / "q_8_7_56_1491_1494_charge_current_closure_declaration_gate_metrics.json"
CHARGE_CLOSURE_EVAL = PUBLIC_OUT / "q_8_7_56_1491_1494_charge_current_closure_numeric_evaluation_metrics.json"
GAP_CLOSEOUT_GATE = PUBLIC_OUT / "q_8_7_56_1495_1498_charge_current_gap_closeout_declaration_gate_metrics.json"
GAP_CLOSEOUT_ROUTE = PUBLIC_OUT / "q_8_7_56_1495_1498_charge_current_gap_closeout_route_sync_metrics.json"
REGISTRY_GATE = PUBLIC_OUT / "q_8_7_56_1499_1502_future_reopen_registry_declaration_gate_metrics.json"
REGISTRY_ROUTE = PUBLIC_OUT / "q_8_7_56_1499_1502_future_reopen_registry_route_sync_metrics.json"
SOURCE_THEOREM_GATE = PUBLIC_OUT / "q_8_7_56_1487_1490_effective_source_theorem_declaration_gate_metrics.json"

SCRIPT_1487 = ROOT / "scripts" / "quantum" / "t2a_1487.py"
SCRIPT_1491 = ROOT / "scripts" / "quantum" / "t2a_1491.py"
SCRIPT_1495 = ROOT / "scripts" / "quantum" / "t2a_1495.py"
SCRIPT_1499 = ROOT / "scripts" / "quantum" / "t2a_1499.py"

CANONICAL_BUNDLE_DIR = PRIVATE_OUT / "expert_review_bundle_20260327_103258"
CANONICAL_BUNDLE_ZIP = PRIVATE_OUT / "expert_review_bundle_20260327_103258.zip"

STEP_TAG = "8.7.56.1503-1506"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor future reopen advice-pack refresh"
STEM = build_compact_artifact_stem(STEP_TAG, "future_reopen_advice_pack", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_future_reopen_ordering_registry_completed"
BRANCH_CLASS = "vector_qball_form_factor_future_reopen_advice_pack_refresh_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_expert_response_reopen_input_handoff"
NEXT_ROUTE = "8.7.56.1507"

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


# 関数: expert-facing summary 文を作る。

def build_advice_pack_text() -> str:
    """Build one concise expert-facing advice-pack summary sentence."""
    return (
        "Current pack remains Case C honest partial: retained scalar exact-profile candidate stays strong at "
        "F_exact(q_theory)=0.2998913524347805 and alpha_exact(q_theory)=0.00715678583937324, restored exact vector "
        "branch stays nontrivial but blind fixed-q observable remains negative at F(q_theory)=-0.083735013520183 "
        "and alpha(q_theory)=0.0005579616187042394, exact charge-current / Noether-current closure is still absent, "
        "proxy signed density stays proxy-only, physical reject stays false, and the future reopen ordering remains "
        "exact_charge_current_noether_closure_reopen, effective_source_theorem_reopen, "
        "observable_dictionary_exact_charge_current_bridge."
    )


# 関数: bundle README を返す。

def bundle_readme_text() -> str:
    """Return the canonical README text for the refreshed advice pack bundle."""
    return (
        "Expert review bundle\n\n"
        "Purpose\n"
        "- Current route: Trial-2 numeric alpha vector-Qball future reopen advice-pack refresh.\n"
        "- Current official disposition: Case C honest partial.\n"
        "- Physical reject required: false.\n"
        "- This bundle asks which reopen surface should be tackled first without diluting the retained scalar strong candidate.\n\n"
        "Frozen reopen ordering\n"
        f"- Primary: {PRIMARY_TRIGGER}\n"
        f"- Secondary: {SECONDARY_TRIGGER}\n"
        f"- Reserve: {RESERVE_TRIGGER}\n\n"
        "Retained anchors\n"
        "- Scalar strong candidate: F_exact(q_theory)=0.2998913524347805, alpha_exact(q_theory)=0.00715678583937324.\n"
        "- Restored exact vector branch: F(q_theory)=-0.083735013520183, alpha(q_theory)=0.0005579616187042394.\n"
    )


# 関数: bundle note を返す。

def bundle_note_text() -> str:
    """Return the canonical bundle note text for the refreshed advice pack."""
    return (
        "Future reopen advice-pack note\n\n"
        "Frozen reading\n"
        "- Retained scalar exact-profile candidate stays strong.\n"
        "- Restored exact vector branch stays nontrivial but blind fixed-q observable still fails.\n"
        "- Exact source theorem is still absent.\n"
        "- Exact charge-current / Noether-current closure is still absent.\n"
        "- Proxy signed density stays proxy-only.\n"
        "- Physical reject is not selected.\n\n"
        "Current ask\n"
        "- Is there any hidden exact charge-current / Noether-current closure surface under the current public pack?\n"
        "- If not, what minimal closure theorem is needed before source-theorem or observable-dictionary work can honestly reopen?\n"
    )


# 関数: review questions を返す。

def questions_text() -> str:
    """Return the canonical review-question text for the refreshed advice pack."""
    return (
        "Questions for review\n\n"
        "1. Does the current public pack contain any hidden exact charge-current / Noether-current closure for the restored exact vector branch?\n"
        "2. If not, what minimal closure theorem would promote proxy signed density into an exact charge current?\n"
        "3. Is the current reopen ordering honest: exact_charge_current_noether_closure_reopen -> effective_source_theorem_reopen -> observable_dictionary_exact_charge_current_bridge?\n"
        "4. Does the remote crossing q/m0 = 0.1255441136164974 have any honest role before exact charge-current closure exists?\n"
        "5. Is the wording \"Case C honest partial / retained scalar strong candidate / physical reject not selected\" the right expert-facing summary?\n"
    )


# 関数: bundle manifest を返す。

def manifest_text(copied_sources: list[Path]) -> str:
    """Return the manifest text for the refreshed advice pack bundle."""
    lines = [
        "Future reopen advice-pack bundle manifest",
        f"Generated: {now_iso()}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# 関数: canonical bundle を rebuilt する。

def rebuild_bundle(files_to_copy: list[Path]) -> dict[str, object]:
    """Rebuild the canonical expert bundle in place with the refreshed advice pack."""
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


# 関数: `.1503-.1506` を実行する。

def main() -> None:
    """Execute the future reopen advice-pack refresh branch."""
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
        UNIFIED_PLAN,
        NEXT_STEPS,
        NEXT_ACTION,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
        CHARGE_CLOSURE_GATE,
        CHARGE_CLOSURE_EVAL,
        GAP_CLOSEOUT_GATE,
        GAP_CLOSEOUT_ROUTE,
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        SOURCE_THEOREM_GATE,
        SCRIPT_1487,
        SCRIPT_1491,
        SCRIPT_1495,
        SCRIPT_1499,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    reopen_advice_text = read_text(REOPEN_ADVICE)
    case_gamma_text = read_text(CASE_GAMMA_ADVICE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    charge_closure_gate = read_json(CHARGE_CLOSURE_GATE)["summary"]
    charge_closure_eval_payload = read_json(CHARGE_CLOSURE_EVAL)
    charge_closure_eval = charge_closure_eval_payload["summary"]
    charge_closure_retained_state = charge_closure_eval_payload["evidence"]["retained_numeric_state"]
    gap_closeout_gate = read_json(GAP_CLOSEOUT_GATE)["summary"]
    gap_closeout_route = read_json(GAP_CLOSEOUT_ROUTE)["summary"]
    registry_gate = read_json(REGISTRY_GATE)["summary"]
    registry_route = read_json(REGISTRY_ROUTE)["summary"]
    source_theorem_gate = read_json(SOURCE_THEOREM_GATE)["summary"]

    status_advice_pack = hit(status_text, "future reopen advice-pack refresh")
    roadmap_advice_pack = hit(roadmap_text, "`8.7.56.1503-.1506`")
    current_problem_advice_pack = hit(current_problem_text, "future reopen advice-pack refresh")
    current_status_advice_pack = hit(current_status_text, "future reopen advice-pack refresh")
    unified_advice_pack = hit(unified_roadmap_text, "future reopen advice-pack refresh")
    reopen_advice_case_c = hit(reopen_advice_text, "Case C honest partial")
    reopen_advice_primary = hit(reopen_advice_text, PRIMARY_TRIGGER)
    reopen_advice_secondary = hit(reopen_advice_text, SECONDARY_TRIGGER)
    reopen_advice_reserve = hit(reopen_advice_text, RESERVE_TRIGGER)
    reopen_advice_proxy = hit(reopen_advice_text, "proxy_signed_density_only")
    case_gamma_case = hit(case_gamma_text, "Case γ")
    case_gamma_primary = hit(case_gamma_text, PRIMARY_TRIGGER)
    part1_noether = hit(part1_text, "Noether保存則")
    part3a_identity = hit(part3a_text, "Q-ball Noether charge = adopted U(1) charge")
    part5_advice_pack = hit(part5_text, "future reopen advice-pack refresh")
    part5_reject_false = hit(part5_text, "physical_reject_required = false")

    inventory_ready = all(
        item is not None
        for item in (
            status_advice_pack,
            roadmap_advice_pack,
            current_problem_advice_pack,
            current_status_advice_pack,
            unified_advice_pack,
            reopen_advice_case_c,
            reopen_advice_primary,
            reopen_advice_secondary,
            reopen_advice_reserve,
            case_gamma_case,
            case_gamma_primary,
            part1_noether,
            part3a_identity,
            part5_advice_pack,
            part5_reject_false,
        )
    )

    prior_registry_available = bool(
        registry_gate.get("reopen_ordering_registry_ready", False)
        and registry_route.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
    )
    closure_fail_retained = bool(gap_closeout_gate.get("closure_fail_retained", False))
    proxy_signed_density_only_retained = bool(
        charge_closure_gate.get("proxy_signed_density_only", False)
        and charge_closure_gate.get("exact_charge_current_noether_closure_available", True) is False
    )
    reopen_ordering_retained = bool(
        registry_gate.get("primary_future_reopen_trigger") == PRIMARY_TRIGGER
        and registry_gate.get("secondary_future_reopen_trigger") == SECONDARY_TRIGGER
        and registry_gate.get("reserve_future_reopen_trigger") == RESERVE_TRIGGER
    )
    physical_reject_not_selected = bool(
        not registry_gate.get("physical_reject_required", True)
        and not gap_closeout_gate.get("physical_reject_required", True)
    )
    retained_scalar_strong_candidate_retained = bool(
        registry_gate.get("retained_scalar_strong_candidate_retained", False)
        and charge_closure_retained_state.get("phase1_equivalent_F_at_q_theory") == -0.083735013520183
    )
    wording_concise = len(build_advice_pack_text().split()) <= 80
    expert_facing_wording_honest = all(
        [
            inventory_ready,
            prior_registry_available,
            closure_fail_retained,
            proxy_signed_density_only_retained,
            reopen_ordering_retained,
            retained_scalar_strong_candidate_retained,
            physical_reject_not_selected,
            source_theorem_gate.get("exact_source_theorem_derived", False) is False,
            reopen_advice_proxy is not None,
        ]
    )
    advice_pack_refresh_ready = bool(expert_facing_wording_honest and wording_concise)
    expert_response_reopen_input_handoff_required = bool(advice_pack_refresh_ready)

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "future reopen advice-pack inventory ready",
            truth(inventory_ready),
            "The advice pack is only honest after the registry outputs, current notes, advice notes, and Part I / Part III-A / Part V wording coexist in one pack.",
        ),
        row(
            "prior_registry_available",
            "pass" if prior_registry_available else "reject",
            "future reopen ordering registry available for advice-pack refresh",
            truth(prior_registry_available),
            "The advice pack can refresh only after the reopen ordering has already been frozen in machine-readable form.",
        ),
        row(
            "closure_fail_retained",
            "pass" if closure_fail_retained else "reject",
            "exact charge-current / Noether-current closure fail retained",
            truth(closure_fail_retained),
            "The refreshed advice pack must keep the current-pack closure fail explicit instead of diluting it into a generic hold state.",
        ),
        row(
            "proxy_signed_density_only_retained",
            "pass" if proxy_signed_density_only_retained else "reject",
            "proxy signed density only retained in advice pack",
            truth(proxy_signed_density_only_retained),
            "The missing piece is now localized enough that the proxy-only status must stay explicit in expert-facing wording.",
        ),
        row(
            "reopen_ordering_retained",
            "pass" if reopen_ordering_retained else "reject",
            "future reopen ordering retained in advice pack",
            truth(reopen_ordering_retained),
            "The advice pack is honest only if it preserves the frozen primary / secondary / reserve ordering verbatim.",
        ),
        row(
            "retained_scalar_strong_candidate_retained",
            "pass" if retained_scalar_strong_candidate_retained else "reject",
            "retained scalar strong candidate kept visible in advice pack",
            truth(retained_scalar_strong_candidate_retained),
            "The advice pack must keep the scalar-side strong candidate visible so the current state does not read like a full reject.",
        ),
        row(
            "physical_reject_not_selected",
            "pass" if physical_reject_not_selected else "reject",
            "physical reject not selected after advice-pack refresh",
            truth(physical_reject_not_selected),
            "The refreshed advice pack remains a retained reopen state and must keep physical_reject_required=false.",
        ),
        row(
            "expert_facing_wording_honest",
            "pass" if expert_facing_wording_honest else "reject",
            "expert-facing advice-pack wording honest",
            truth(expert_facing_wording_honest),
            "The refreshed wording is honest only if it keeps Case C, the blind vector no-go, the closure fail, the proxy-only state, and the frozen reopen ordering together.",
        ),
        row(
            "expert_facing_wording_concise",
            "pass" if wording_concise else "reject",
            "expert-facing advice-pack wording concise",
            truth(wording_concise),
            "The final internal wording branch should compress the reopen state into one short advice pack rather than reopening theorem-side detail.",
        ),
        row(
            "expert_response_reopen_input_handoff_required",
            "pass" if expert_response_reopen_input_handoff_required else "reject",
            "expert response / reopen input handoff required",
            truth(expert_response_reopen_input_handoff_required),
            "Once the advice pack is refreshed, the next honest action is expert response / reopen input handoff rather than another internal wording loop.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "primary_sources": display_path(PRIMARY_SOURCES),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "reopen_advice_note": display_path(REOPEN_ADVICE),
            "case_gamma_advice_note": display_path(CASE_GAMMA_ADVICE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
        },
        "prior_metrics": {
            "charge_closure_gate": display_path(CHARGE_CLOSURE_GATE),
            "gap_closeout_gate": display_path(GAP_CLOSEOUT_GATE),
            "gap_closeout_route": display_path(GAP_CLOSEOUT_ROUTE),
            "registry_gate": display_path(REGISTRY_GATE),
            "registry_route": display_path(REGISTRY_ROUTE),
            "source_theorem_gate": display_path(SOURCE_THEOREM_GATE),
        },
        "constants": {
            "primary_future_reopen_trigger": PRIMARY_TRIGGER,
            "secondary_future_reopen_trigger": SECONDARY_TRIGGER,
            "reserve_future_reopen_trigger": RESERVE_TRIGGER,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    common_summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "inventory_ready": inventory_ready,
        "prior_registry_available": prior_registry_available,
        "closure_fail_retained": closure_fail_retained,
        "proxy_signed_density_only_retained": proxy_signed_density_only_retained,
        "reopen_ordering_retained": reopen_ordering_retained,
        "retained_scalar_strong_candidate_retained": retained_scalar_strong_candidate_retained,
        "expert_facing_wording_honest": expert_facing_wording_honest,
        "expert_facing_wording_concise": wording_concise,
        "future_reopen_advice_pack_refresh_ready": advice_pack_refresh_ready,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }
    common_decision = {
        "overall_status": f"{BRANCH_CLASS}_documented",
        "branch_completed": advice_pack_refresh_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    inventory_evidence = {
        "doc_hits": {
            "status_advice_pack": status_advice_pack,
            "roadmap_advice_pack": roadmap_advice_pack,
            "current_problem_advice_pack": current_problem_advice_pack,
            "current_status_advice_pack": current_status_advice_pack,
            "unified_advice_pack": unified_advice_pack,
            "reopen_advice_case_c": reopen_advice_case_c,
            "reopen_advice_primary": reopen_advice_primary,
            "reopen_advice_secondary": reopen_advice_secondary,
            "reopen_advice_reserve": reopen_advice_reserve,
            "reopen_advice_proxy": reopen_advice_proxy,
            "case_gamma_case": case_gamma_case,
            "case_gamma_primary": case_gamma_primary,
            "part1_noether": part1_noether,
            "part3a_identity": part3a_identity,
            "part5_advice_pack": part5_advice_pack,
            "part5_reject_false": part5_reject_false,
        }
    }
    audit_evidence = {
        "advice_pack_text": build_advice_pack_text(),
        "carry_over": {
            "charge_closure_gate": charge_closure_gate,
            "gap_closeout_gate": gap_closeout_gate,
            "registry_gate": registry_gate,
            "source_theorem_gate": source_theorem_gate,
        },
        "retained_numeric_state": {
            "phase1_equivalent_F_at_q_theory": charge_closure_retained_state["phase1_equivalent_F_at_q_theory"],
            "phase1_equivalent_alpha_at_q_theory": charge_closure_retained_state["phase1_equivalent_alpha_at_q_theory"],
            "phase1_equivalent_max_abs_ratio": charge_closure_retained_state["phase1_equivalent_max_abs_ratio"],
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
        UNIFIED_PLAN,
        NEXT_STEPS,
        NEXT_ACTION,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
        SCRIPT_1487,
        SCRIPT_1491,
        SCRIPT_1495,
        SCRIPT_1499,
        ROOT / inventory_paths["json"],
        ROOT / audit_paths["json"],
        ROOT / gate_paths["json"],
        CHARGE_CLOSURE_GATE,
        GAP_CLOSEOUT_GATE,
        GAP_CLOSEOUT_ROUTE,
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        SOURCE_THEOREM_GATE,
    ]
    bundle_stats = rebuild_bundle(refresh_candidates)

    route_summary = {
        **common_summary,
        "future_reopen_advice_pack_refresh_completed": advice_pack_refresh_ready,
        "advice_pack_bundle_sync_complete": advice_pack_refresh_ready,
        "share_pack_bundle_dir": bundle_stats["bundle_dir"],
        "share_pack_bundle_zip": bundle_stats["bundle_zip"],
        "share_pack_bundle_refresh_count": bundle_stats["copied_count"],
        "share_pack_staging_file_count": bundle_stats["staging_file_count"],
        "share_pack_zip_file_count": bundle_stats["zip_file_count"],
        "expert_response_reopen_input_handoff_required": expert_response_reopen_input_handoff_required,
        "numeric_state_changed_by_current_branch": False,
        "route_state_changed_by_current_branch": True,
    }
    route_evidence = {
        "bundle_stats": bundle_stats,
        "advice_pack_text": build_advice_pack_text(),
        "carry_over": {
            "charge_closure_gate": charge_closure_gate,
            "gap_closeout_gate": gap_closeout_gate,
            "registry_gate": registry_gate,
            "registry_route": registry_route,
            "source_theorem_gate": source_theorem_gate,
        },
    }
    write_artifact(
        "route_sync",
        payload(STEP_TAG, STEP_NAME, inputs, rows, route_summary, common_decision, route_evidence),
    )

    print(f"[ok] wrote compact artifacts for {STEP_TAG}: {STEM}")


if __name__ == "__main__":
    main()
