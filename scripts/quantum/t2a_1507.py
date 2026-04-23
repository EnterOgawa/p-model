#!/usr/bin/env python3
"""Generate 8.7.56.1507-.1510 expert response / reopen input handoff artifacts.

This branch closes the internal wording loop after the refreshed advice pack.
It freezes the handoff surface for external expert response or reopen input and
reorganizes the roadmap so the next branch is conditional on new external input.
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
ADVICE_INV = PUBLIC_OUT / "q_8_7_56_1503_1506_future_reopen_advice_pack_inventory_metrics.json"
ADVICE_AUDIT = PUBLIC_OUT / "q_8_7_56_1503_1506_future_reopen_advice_pack_audit_metrics.json"
ADVICE_GATE = PUBLIC_OUT / "q_8_7_56_1503_1506_future_reopen_advice_pack_declaration_gate_metrics.json"
ADVICE_ROUTE = PUBLIC_OUT / "q_8_7_56_1503_1506_future_reopen_advice_pack_route_sync_metrics.json"

SCRIPT_1487 = ROOT / "scripts" / "quantum" / "t2a_1487.py"
SCRIPT_1491 = ROOT / "scripts" / "quantum" / "t2a_1491.py"
SCRIPT_1495 = ROOT / "scripts" / "quantum" / "t2a_1495.py"
SCRIPT_1499 = ROOT / "scripts" / "quantum" / "t2a_1499.py"
SCRIPT_1503 = ROOT / "scripts" / "quantum" / "t2a_1503.py"

CANONICAL_BUNDLE_DIR = PRIVATE_OUT / "expert_review_bundle_20260327_103258"
CANONICAL_BUNDLE_ZIP = PRIVATE_OUT / "expert_review_bundle_20260327_103258.zip"

STEP_TAG = "8.7.56.1507-1510"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor expert response / reopen input handoff"
STEM = build_compact_artifact_stem(STEP_TAG, "expert_reopen_handoff", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_future_reopen_advice_pack_refresh_completed"
BRANCH_CLASS = "vector_qball_form_factor_expert_response_reopen_input_handoff_completed"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_conditional_expert_response_reopen_input_assimilation"
NEXT_ROUTE = "8.7.56.1511"
NEXT_ROUTE_ACTIVATION_CONDITION = "new external expert response or reopen input arrives"

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


# 関数: concise な handoff summary 文を作る。

def build_handoff_text() -> str:
    """Build one concise expert-response handoff summary sentence."""
    return (
        "Current pack is frozen for external input: closure fail and proxy-only signed density remain retained, "
        "the scalar strong candidate stays at F_exact(q_theory)=0.2998913524347805 and alpha_exact(q_theory)=0.00715678583937324, "
        "the restored exact vector branch still gives blind fixed-q no-go at F(q_theory)=-0.083735013520183 and "
        "alpha(q_theory)=0.0005579616187042394, physical reject stays false, and reopen ordering remains "
        "exact_charge_current_noether_closure_reopen, effective_source_theorem_reopen, observable_dictionary_exact_charge_current_bridge."
    )


# 関数: bundle README を返す。

def bundle_readme_text() -> str:
    """Return the canonical README text for the expert-response handoff bundle."""
    return (
        "Expert response / reopen input handoff bundle\n\n"
        "Purpose\n"
        "- Current route: Trial-2 numeric alpha vector-Qball expert response / reopen input handoff.\n"
        "- Current official disposition: Case C honest partial.\n"
        "- Physical reject required: false.\n"
        "- This bundle is the final internal handoff surface before any new expert response or reopen input arrives.\n\n"
        "Frozen reopen ordering\n"
        f"- Primary: {PRIMARY_TRIGGER}\n"
        f"- Secondary: {SECONDARY_TRIGGER}\n"
        f"- Reserve: {RESERVE_TRIGGER}\n\n"
        "Retained anchors\n"
        "- Scalar strong candidate: F_exact(q_theory)=0.2998913524347805, alpha_exact(q_theory)=0.00715678583937324.\n"
        "- Restored exact vector branch: F(q_theory)=-0.083735013520183, alpha(q_theory)=0.0005579616187042394.\n\n"
        "Rule\n"
        "- Do not extend the internal wording loop again without new external input.\n"
    )


# 関数: bundle note を返す。

def bundle_note_text() -> str:
    """Return the canonical bundle note text for the handoff bundle."""
    return (
        "Expert response / reopen input handoff note\n\n"
        "Frozen reading\n"
        "- Case C honest partial is retained.\n"
        "- Exact charge-current / Noether-current closure is still absent.\n"
        "- Exact source theorem is still absent.\n"
        "- Proxy signed density stays proxy-only.\n"
        "- Scalar strong candidate stays retained.\n"
        "- Blind fixed-q vector observable stays no-go.\n"
        "- Physical reject is not selected.\n\n"
        "Operational rule\n"
        "- New work should start only when an external expert response or reopen input arrives.\n"
        "- Until then, the reopen ordering above remains frozen.\n"
    )


# 関数: review questions を返す。

def questions_text() -> str:
    """Return the canonical review-question text for the handoff bundle."""
    return (
        "Questions for expert response / reopen input\n\n"
        "1. Does new input open an exact charge-current / Noether-current closure surface for the restored exact vector branch?\n"
        "2. If not, does it open a narrower exact source-theorem surface without closing charge current first?\n"
        "3. If neither opens, does it justify changing the frozen reopen ordering?\n"
        "4. Does any new input promote observable_dictionary_exact_charge_current_bridge above reserve status?\n"
        "5. If no new theorem surface opens, should the current Case C honest partial disposition remain frozen as-is?\n"
    )


# 関数: bundle manifest を返す。

def manifest_text(copied_sources: list[Path]) -> str:
    """Return the manifest text for the handoff bundle."""
    lines = [
        "Expert response / reopen input handoff bundle manifest",
        f"Generated: {now_iso()}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# 関数: canonical bundle を rebuilt する。

def rebuild_bundle(files_to_copy: list[Path]) -> dict[str, object]:
    """Rebuild the canonical expert bundle in place with handoff-facing text."""
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


# 関数: `.1507-.1510` を実行する。

def main() -> None:
    """Execute the expert response / reopen input handoff branch."""
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
        ADVICE_INV,
        ADVICE_AUDIT,
        ADVICE_GATE,
        ADVICE_ROUTE,
        SCRIPT_1487,
        SCRIPT_1491,
        SCRIPT_1495,
        SCRIPT_1499,
        SCRIPT_1503,
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
    charge_closure_retained_state = charge_closure_eval_payload["evidence"]["retained_numeric_state"]
    gap_closeout_gate = read_json(GAP_CLOSEOUT_GATE)["summary"]
    gap_closeout_route = read_json(GAP_CLOSEOUT_ROUTE)["summary"]
    registry_gate = read_json(REGISTRY_GATE)["summary"]
    registry_route = read_json(REGISTRY_ROUTE)["summary"]
    advice_inv = read_json(ADVICE_INV)["summary"]
    advice_audit = read_json(ADVICE_AUDIT)["summary"]
    advice_gate = read_json(ADVICE_GATE)["summary"]
    advice_route = read_json(ADVICE_ROUTE)["summary"]

    status_handoff = hit(status_text, "expert response / reopen input handoff")
    roadmap_handoff = hit(roadmap_text, "`8.7.56.1507-.1510`")
    current_problem_handoff = hit(current_problem_text, "expert response / reopen input handoff")
    current_status_handoff = hit(current_status_text, "expert response / reopen input handoff")
    unified_handoff = hit(unified_roadmap_text, "expert response / reopen input handoff")
    reopen_advice_case_c = hit(reopen_advice_text, "Case C honest partial")
    reopen_advice_primary = hit(reopen_advice_text, PRIMARY_TRIGGER)
    reopen_advice_secondary = hit(reopen_advice_text, SECONDARY_TRIGGER)
    reopen_advice_reserve = hit(reopen_advice_text, RESERVE_TRIGGER)
    case_gamma_case = hit(case_gamma_text, "Case γ")
    part1_noether = hit(part1_text, "Noether保存則")
    part3a_identity = hit(part3a_text, "Q-ball Noether charge = adopted U(1) charge")
    part5_handoff = hit(part5_text, "expert response / reopen input handoff")
    part5_reject_false = hit(part5_text, "physical_reject_required = false")

    inventory_ready = all(
        item is not None
        for item in (
            status_handoff,
            roadmap_handoff,
            current_problem_handoff,
            current_status_handoff,
            unified_handoff,
            reopen_advice_case_c,
            reopen_advice_primary,
            reopen_advice_secondary,
            reopen_advice_reserve,
            case_gamma_case,
            part1_noether,
            part3a_identity,
            part5_handoff,
            part5_reject_false,
        )
    )

    prior_advice_pack_available = bool(
        advice_inv.get("inventory_ready", False)
        and advice_audit.get("expert_facing_wording_honest", False)
        and advice_gate.get("future_reopen_advice_pack_refresh_ready", False)
        and advice_route.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
    )
    share_pack_bundle_available = bool(CANONICAL_BUNDLE_DIR.exists() and CANONICAL_BUNDLE_ZIP.exists())
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
    retained_scalar_strong_candidate_retained = bool(advice_route.get("retained_scalar_strong_candidate_retained", False))
    blind_vector_no_go_retained = bool(charge_closure_retained_state["phase1_equivalent_F_at_q_theory"] == -0.083735013520183)
    physical_reject_not_selected = bool(
        not advice_route.get("physical_reject_required", True)
        and not registry_gate.get("physical_reject_required", True)
        and not gap_closeout_route.get("physical_reject_required", True)
    )
    handoff_text = build_handoff_text()
    handoff_word_count = len(handoff_text.split())
    handoff_concise = handoff_word_count <= 85
    expert_response_reopen_input_handoff_ready = all(
        [
            inventory_ready,
            prior_advice_pack_available,
            share_pack_bundle_available,
            closure_fail_retained,
            proxy_signed_density_only_retained,
            reopen_ordering_retained,
            retained_scalar_strong_candidate_retained,
            blind_vector_no_go_retained,
            physical_reject_not_selected,
            handoff_concise,
        ]
    )
    internal_wording_loop_extension_admissible = False
    external_input_required_before_next_branch = True
    conditional_external_input_assimilation_required = expert_response_reopen_input_handoff_ready

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "expert response / reopen input handoff inventory ready",
            truth(inventory_ready),
            "The handoff pack is only honest after the refreshed advice pack, current notes, bundle, and Part I / Part III-A / Part V wording coexist in one pack.",
        ),
        row(
            "prior_advice_pack_available",
            "pass" if prior_advice_pack_available else "reject",
            "future reopen advice-pack refresh available for handoff",
            truth(prior_advice_pack_available),
            "The handoff route starts only after the refreshed advice pack is already fixed in machine-readable form.",
        ),
        row(
            "share_pack_bundle_available",
            "pass" if share_pack_bundle_available else "reject",
            "canonical share-pack bundle available for handoff",
            truth(share_pack_bundle_available),
            "The handoff route reuses the canonical expert bundle instead of creating another internal wording branch.",
        ),
        row(
            "closure_fail_retained",
            "pass" if closure_fail_retained else "reject",
            "exact charge-current / Noether-current closure fail retained in handoff",
            truth(closure_fail_retained),
            "The handoff remains honest only if the current-pack closure fail stays explicit.",
        ),
        row(
            "proxy_signed_density_only_retained",
            "pass" if proxy_signed_density_only_retained else "reject",
            "proxy signed density only retained in handoff",
            truth(proxy_signed_density_only_retained),
            "The missing surface is localized enough that the proxy-only status must stay explicit at handoff time.",
        ),
        row(
            "reopen_ordering_retained",
            "pass" if reopen_ordering_retained else "reject",
            "frozen reopen ordering retained in handoff",
            truth(reopen_ordering_retained),
            "The handoff is honest only if it preserves the primary / secondary / reserve ordering verbatim.",
        ),
        row(
            "retained_scalar_strong_candidate_retained",
            "pass" if retained_scalar_strong_candidate_retained else "reject",
            "retained scalar strong candidate kept visible in handoff",
            truth(retained_scalar_strong_candidate_retained),
            "The scalar-side strong candidate must stay visible so the route does not read like a full reject.",
        ),
        row(
            "blind_vector_no_go_retained",
            "pass" if blind_vector_no_go_retained else "reject",
            "blind fixed-q vector no-go retained in handoff",
            truth(blind_vector_no_go_retained),
            "The negative fixed-q vector result stays part of the handoff surface and is not diluted into a generic hold state.",
        ),
        row(
            "expert_response_reopen_input_handoff_ready",
            "pass" if expert_response_reopen_input_handoff_ready else "reject",
            "expert response / reopen input handoff ready",
            truth(expert_response_reopen_input_handoff_ready),
            "Once the advice pack is refreshed, the next honest action is handoff to external input rather than another internal loop.",
        ),
        row(
            "external_input_required_before_next_branch",
            "pass" if external_input_required_before_next_branch else "reject",
            "new external input required before the next branch",
            truth(external_input_required_before_next_branch),
            "The next branch should activate only when a real expert response or reopen input arrives.",
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
            "charge_closure_eval": display_path(CHARGE_CLOSURE_EVAL),
            "gap_closeout_gate": display_path(GAP_CLOSEOUT_GATE),
            "gap_closeout_route": display_path(GAP_CLOSEOUT_ROUTE),
            "registry_gate": display_path(REGISTRY_GATE),
            "registry_route": display_path(REGISTRY_ROUTE),
            "advice_inventory": display_path(ADVICE_INV),
            "advice_audit": display_path(ADVICE_AUDIT),
            "advice_gate": display_path(ADVICE_GATE),
            "advice_route": display_path(ADVICE_ROUTE),
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
        "inventory_ready": inventory_ready,
        "prior_advice_pack_available": prior_advice_pack_available,
        "closure_fail_retained": closure_fail_retained,
        "proxy_signed_density_only_retained": proxy_signed_density_only_retained,
        "reopen_ordering_retained": reopen_ordering_retained,
        "retained_scalar_strong_candidate_retained": retained_scalar_strong_candidate_retained,
        "blind_vector_no_go_retained": blind_vector_no_go_retained,
        "expert_response_reopen_input_handoff_ready": expert_response_reopen_input_handoff_ready,
        "expert_handoff_wording_honest": expert_response_reopen_input_handoff_ready,
        "expert_handoff_wording_concise": handoff_concise,
        "external_input_required_before_next_branch": external_input_required_before_next_branch,
        "internal_wording_loop_extension_admissible": internal_wording_loop_extension_admissible,
        "physical_reject_required": False,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
    }
    common_decision = {
        "overall_status": f"{BRANCH_CLASS}_documented",
        "branch_completed": expert_response_reopen_input_handoff_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    inventory_evidence = {
        "doc_hits": {
            "status_handoff": status_handoff,
            "roadmap_handoff": roadmap_handoff,
            "current_problem_handoff": current_problem_handoff,
            "current_status_handoff": current_status_handoff,
            "unified_handoff": unified_handoff,
            "reopen_advice_case_c": reopen_advice_case_c,
            "reopen_advice_primary": reopen_advice_primary,
            "reopen_advice_secondary": reopen_advice_secondary,
            "reopen_advice_reserve": reopen_advice_reserve,
            "case_gamma_case": case_gamma_case,
            "part1_noether": part1_noether,
            "part3a_identity": part3a_identity,
            "part5_handoff": part5_handoff,
            "part5_reject_false": part5_reject_false,
        }
    }
    audit_evidence = {
        "handoff_text": handoff_text,
        "carry_over": {
            "charge_closure_gate": charge_closure_gate,
            "gap_closeout_gate": gap_closeout_gate,
            "registry_gate": registry_gate,
            "advice_gate": advice_gate,
            "advice_route": advice_route,
        },
        "retained_numeric_state": {
            "phase1_equivalent_F_at_q_theory": charge_closure_retained_state["phase1_equivalent_F_at_q_theory"],
            "phase1_equivalent_alpha_at_q_theory": charge_closure_retained_state["phase1_equivalent_alpha_at_q_theory"],
            "phase1_equivalent_max_abs_ratio": charge_closure_retained_state["phase1_equivalent_max_abs_ratio"],
        },
        "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
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
        SCRIPT_1503,
        Path(__file__),
        CHARGE_CLOSURE_GATE,
        GAP_CLOSEOUT_GATE,
        GAP_CLOSEOUT_ROUTE,
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        ADVICE_INV,
        ADVICE_AUDIT,
        ADVICE_GATE,
        ADVICE_ROUTE,
        ROOT / inventory_paths["json"],
        ROOT / audit_paths["json"],
        ROOT / gate_paths["json"],
    ]
    bundle_stats = rebuild_bundle(refresh_candidates)

    route_summary = {
        **common_summary,
        "expert_response_reopen_input_handoff_completed": expert_response_reopen_input_handoff_ready,
        "advice_pack_bundle_sync_complete": expert_response_reopen_input_handoff_ready,
        "share_pack_bundle_dir": bundle_stats["bundle_dir"],
        "share_pack_bundle_zip": bundle_stats["bundle_zip"],
        "share_pack_bundle_refresh_count": bundle_stats["copied_count"],
        "share_pack_staging_file_count": bundle_stats["staging_file_count"],
        "share_pack_zip_file_count": bundle_stats["zip_file_count"],
        "conditional_external_input_assimilation_required": conditional_external_input_assimilation_required,
        "next_route_activation_condition": NEXT_ROUTE_ACTIVATION_CONDITION,
        "numeric_state_changed_by_current_branch": False,
        "route_state_changed_by_current_branch": True,
    }
    route_evidence = {
        "bundle_stats": bundle_stats,
        "handoff_text": handoff_text,
        "carry_over": {
            "charge_closure_eval": charge_closure_eval_payload["summary"],
            "gap_closeout_route": gap_closeout_route,
            "registry_route": registry_route,
            "advice_route": advice_route,
        },
    }
    write_artifact(
        "route_sync",
        payload(STEP_TAG, STEP_NAME, inputs, rows, route_summary, common_decision, route_evidence),
    )

    print(f"[ok] wrote compact artifacts for {STEP_TAG}: {STEM}")


if __name__ == "__main__":
    main()
