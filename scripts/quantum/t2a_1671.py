#!/usr/bin/env python3
"""Generate 8.7.56.1671-.1674 fallback closeout advice-pack refresh artifacts.

This branch does not reopen the frozen-action computation pack.
`.1667-.1670` already froze the honest closeout:

1. same-level fallback family exhausted,
2. same-level rescue extension inadmissible,
3. new action-level structure becomes the primary reopen surface,
4. future external input remains reserve only.

The job of `.1671-.1674` is therefore operational and expert-facing:

- refresh a compact advice pack,
- keep scalar-leaning evidence-only surfaces visible,
- move the next official route to conditional reactivation only,
- refuse any new same-level rescue lane under the current pack.
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
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
PRIOR_RESPONSE = ROOT / "doc" / "quantum" / "51_trial2_numeric_alpha_vector_qball_branch_local_nonlinear_response.md"
LOCAL_RESPONSE = (
    ROOT
    / "doc"
    / "quantum"
    / "52_trial2_numeric_alpha_vector_qball_fallback_closeout_advice_pack_response.md"
)
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
REGISTRY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1667_1670_fallback_closeout_registry_declaration_gate_metrics.json"
)
REGISTRY_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1667_1670_fallback_closeout_registry_route_sync_metrics.json"
)
SCRIPT_SELF = Path(__file__).resolve()

STEP_TAG = "8.7.56.1671-1674"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fallback closeout "
    "advice-pack refresh"
)
STEM = build_compact_artifact_stem(STEP_TAG, "fb_closeout_advice_pack", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_current_pack_fallback_family_closeout_"
    "reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_fallback_closeout_advice_pack_refresh_completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_new_action_"
    "level_structure_or_external_input_reactivation"
)
NEXT_ROUTE = "8.7.56.1675"
NEXT_BRANCH = "8.7.56.1675-.1678"

PRIMARY_REOPEN = "genuinely_new_action_level_structure_beyond_current_frozen_action_pack"
SECONDARY_REOPEN = (
    "exact_constitutive_map_or_branch_local_full_nonlinear_energy_density_"
    "reopen_after_pack_update"
)
RESERVE_REOPEN = "future_external_input_or_expert_input_guiding_new_primary_surface"

SCALAR_ALPHA = 0.00715678583937324
ENERGY_ALPHA = 0.0005422361373947313
PROJECTED_ALPHA = 0.0005600186431488893
ELECTRIC_LIKE_ALPHA = 0.004692984339643002
NOTE_GRADIENT_ALPHA = 0.0047372462907781755

BUNDLE_DIR = PRIVATE_OUT / "fb_closeout_pack_20260328"
BUNDLE_ZIP = PRIVATE_OUT / "fb_closeout_pack_20260328.zip"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検証する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
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


# 関数: 表示用の相対パスへ変換する。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

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


# 関数: JSON / CSV 成果物を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0 / 1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: UTF-8 テキストを1本書き出す。

def write_text(path: Path, text: str) -> Path:
    """Write one UTF-8 text file and return the same path."""
    path.write_text(text, encoding="utf-8")
    return path


# 関数: bundle README を返す。

def bundle_readme_text() -> str:
    """Return the canonical README text for the refreshed fallback closeout pack."""
    return (
        "Fallback closeout advice pack\n\n"
        "Current official read\n"
        "- current-pack same-level fallback family exhausted\n"
        "- same-level rescue extension inadmissible\n"
        "- retained scalar exact alpha: 0.00715678583937324\n"
        "- official energy-core alpha: 0.0005422361373947313\n"
        "- official projected-kernel alpha: 0.0005600186431488893\n"
        "- physical reject required: false\n\n"
        "Reopen ordering\n"
        "- primary: genuinely new action-level structure beyond current frozen-action pack\n"
        "- secondary: exact constitutive-map or branch-local full nonlinear energy-density reopen after pack update\n"
        "- reserve: future external input or expert input guiding new primary surface\n\n"
        "Operational next route\n"
        "- 8.7.56.1675-.1678 conditional new action-level structure / external-input reactivation\n"
        "- trigger only if a genuinely new canonical surface or genuinely new external input appears\n"
    )


# 関数: bundle note を返す。

def bundle_note_text() -> str:
    """Return the canonical internal note text for this advice pack."""
    return (
        "The current frozen-action pack has exhausted every same-level fallback lane.\n"
        "No additional density / constitutive-map / nonlinear-energy / projected-kernel /\n"
        "branch-selection rescue variant is admissible under the same pack.\n\n"
        "Scalar-leaning evidence-only surfaces remain retained:\n"
        f"- electric-like alpha = {ELECTRIC_LIKE_ALPHA}\n"
        f"- note-gradient alpha = {NOTE_GRADIENT_ALPHA}\n\n"
        "These surfaces do not canonically promote and therefore do not reopen the pack.\n"
    )


# 関数: pack summary を返す。

def pack_summary_text() -> str:
    """Return the concise pack summary text."""
    return (
        "This pack synchronizes the exhausted fallback family and its reopen ordering "
        "for expert-facing handoff. The next official branch is conditional "
        "reactivation only; no same-level internal rescue lane remains active."
    )


# 関数: bundle manifest を返す。

def manifest_text(copied_sources: list[Path]) -> str:
    """Return the bundle manifest text."""
    lines = [
        "Fallback closeout advice-pack manifest",
        f"Generated: {now_iso()}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# 関数: advice-pack bundle を再生成する。

def refresh_bundle(copied_sources: list[Path]) -> dict[str, object]:
    """Rebuild the compact expert-facing bundle for this advice-pack refresh."""
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    staged_paths: list[Path] = []
    for source in copied_sources:
        target = BUNDLE_DIR / source.name
        shutil.copy2(source, target)
        staged_paths.append(target)

    staged_paths.append(write_text(BUNDLE_DIR / "README.txt", bundle_readme_text()))
    staged_paths.append(write_text(BUNDLE_DIR / "BUNDLE_NOTE.txt", bundle_note_text()))
    staged_paths.append(write_text(BUNDLE_DIR / "PACK_SUMMARY.txt", pack_summary_text()))
    staged_paths.append(
        write_text(BUNDLE_DIR / "BUNDLE_MANIFEST.txt", manifest_text(copied_sources))
    )

    if BUNDLE_ZIP.exists():
        BUNDLE_ZIP.unlink()

    with zipfile.ZipFile(BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for staged in sorted(staged_paths):
            handle.write(staged, arcname=staged.name)

    with zipfile.ZipFile(BUNDLE_ZIP, "r") as handle:
        zip_count = len(handle.namelist())

    return {
        "bundle_dir": display_path(BUNDLE_DIR),
        "bundle_zip": display_path(BUNDLE_ZIP),
        "copied_count": len(copied_sources),
        "staging_file_count": len(staged_paths),
        "zip_file_count": zip_count,
    }


# 関数: `.1671-.1674` を実行する。

def main() -> None:
    """Execute the fallback closeout advice-pack refresh branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PRIOR_RESPONSE,
        LOCAL_RESPONSE,
        PART5,
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        SCRIPT_SELF,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)
    local_response_text = read_text(LOCAL_RESPONSE)
    ai_context = read_json(AI_CONTEXT)
    registry_gate = read_json(REGISTRY_GATE)
    registry_route = read_json(REGISTRY_ROUTE)

    copied_sources = [
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PRIOR_RESPONSE,
        LOCAL_RESPONSE,
        PART5,
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        SCRIPT_SELF,
    ]
    bundle = refresh_bundle(copied_sources)

    prior_summary = registry_gate["summary"]
    route_summary = registry_route["summary"]

    inputs = {
        "status": display_path(STATUS),
        "roadmap": display_path(ROADMAP),
        "ai_context": display_path(AI_CONTEXT),
        "work_history_recent": display_path(WORK_HISTORY_RECENT),
        "current_problem": display_path(CURRENT_PROBLEM),
        "current_status": display_path(CURRENT_STATUS),
        "unified_roadmap": display_path(UNIFIED_ROADMAP),
        "prior_response": display_path(PRIOR_RESPONSE),
        "local_response": display_path(LOCAL_RESPONSE),
        "part5": display_path(PART5),
        "registry_gate": display_path(REGISTRY_GATE),
        "registry_route": display_path(REGISTRY_ROUTE),
        "script": display_path(SCRIPT_SELF),
    }

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "fallback closeout advice-pack refresh"),
            hit(roadmap_text, "8.7.56.1671-.1674"),
            hit(current_problem_text, "fallback closeout advice-pack refresh"),
            hit(current_status_text, "fallback closeout advice-pack refresh"),
            hit(unified_text, ".1671-.1674"),
            hit(part5_text, ".1671-.1674"),
        )
    )
    prior_registry_retained = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("same_level_fallback_family_exhausted", False)
        and not prior_summary.get("same_level_rescue_lane_extension_admissible", True)
    )
    reopen_ordering_retained = bool(
        prior_summary.get("primary_reopen_surface") == PRIMARY_REOPEN
        and prior_summary.get("secondary_reopen_surface") == SECONDARY_REOPEN
        and prior_summary.get("reserve_reopen_surface") == RESERVE_REOPEN
    )
    evidence_only_surfaces_retained = bool(
        abs(prior_summary.get("electric_like_component_alpha_at_q_theory", 0.0) - ELECTRIC_LIKE_ALPHA) < 1e-18
        and abs(prior_summary.get("note_gradient_alpha_at_q_theory", 0.0) - NOTE_GRADIENT_ALPHA) < 1e-18
        and prior_summary.get("noncanonical_scalar_leaning_evidence_retained", False)
    )
    conditional_reactivation_only = bool(
        route_summary.get("selected_followup_route") == NEXT_ROUTE_NAME
        and route_summary.get("selected_followup_route_or_none") == NEXT_ROUTE
    )
    advice_pack_ready = bool(
        inventory_ready
        and prior_registry_retained
        and reopen_ordering_retained
        and evidence_only_surfaces_retained
        and conditional_reactivation_only
    )

    inventory_rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "fallback closeout advice-pack inventory ready",
            truth(inventory_ready),
            "Status, roadmap, current notes, unified roadmap, and Part V all point to the same `.1671-.1674` refresh branch.",
        ),
        row(
            "inventory_prior_registry_retained",
            "pass" if prior_registry_retained else "reject",
            "prior fallback closeout registry retained",
            truth(prior_registry_retained),
            "The advice-pack refresh is only honest if the prior `.1667-.1670` registry remains authoritative.",
        ),
        row(
            "inventory_reopen_ordering_retained",
            "pass" if reopen_ordering_retained else "reject",
            "reopen ordering retained",
            truth(reopen_ordering_retained),
            "Primary / secondary / reserve reopen surfaces must carry over without relabeling.",
        ),
        row(
            "inventory_bundle_sources",
            "pass",
            "copied_source_count",
            float(bundle["copied_count"]),
            "The compact advice pack copies only the required state, roadmap, response, and metrics sources.",
        ),
    ]

    audit_rows = [
        row(
            "audit_same_level_exhausted",
            "pass" if prior_summary.get("same_level_fallback_family_exhausted", False) else "reject",
            "same-level fallback family exhausted",
            truth(prior_summary.get("same_level_fallback_family_exhausted", False)),
            "No new same-level rescue lane remains under the current frozen-action pack.",
        ),
        row(
            "audit_extension_inadmissible",
            "pass" if not prior_summary.get("same_level_rescue_lane_extension_admissible", True) else "reject",
            "same-level rescue extension inadmissible",
            truth(not prior_summary.get("same_level_rescue_lane_extension_admissible", True)),
            "The refresh freezes the fact that same-level extension is no longer admissible.",
        ),
        row(
            "audit_scalar_leaning_evidence",
            "pass" if evidence_only_surfaces_retained else "reject",
            "noncanonical scalar-leaning evidence retained",
            truth(evidence_only_surfaces_retained),
            "Electric-like and note-gradient surfaces remain visible as evidence-only hints and are not promoted.",
        ),
        row(
            "audit_conditional_reactivation_only",
            "pass" if conditional_reactivation_only else "reject",
            "conditional reactivation only",
            truth(conditional_reactivation_only),
            "After refresh, the next official route is conditional reactivation only; nothing active remains under the same pack.",
        ),
        row(
            "audit_future_input_side_lane",
            "pass",
            "future external input retained as reserve side lane",
            1.0,
            "Future external input remains useful but only as reserve guidance for a new primary surface.",
        ),
        row(
            "audit_bundle_ready",
            "pass" if advice_pack_ready else "reject",
            "expert-facing bundle ready",
            truth(advice_pack_ready),
            "The compact bundle is honest only if the exhausted-family read and reopen ordering survive unchanged.",
        ),
    ]

    declaration_rows = [
        row(
            "gate_branch_completed",
            "pass" if advice_pack_ready else "reject",
            "fallback closeout advice-pack refresh completed",
            truth(advice_pack_ready),
            "This branch is complete once the exhausted-family closeout has been synchronized into an expert-facing pack.",
        ),
        row(
            "gate_selected_next_generation_route",
            "pass",
            "recommended next route",
            1675.0,
            "The next official route is conditional new action-level structure / external-input reactivation.",
        ),
        row(
            "gate_same_pack_retry_stopped",
            "pass",
            "same-pack retry stopped",
            1.0,
            "No additional same-level rescue trial remains active after this refresh.",
        ),
        row(
            "gate_physical_reject",
            "pass",
            "physical reject required",
            0.0,
            "Physical reject remains false; the closeout is route-local to the current pack.",
        ),
        row(
            "gate_bundle_counts",
            "pass",
            "bundle zip file count",
            float(bundle["zip_file_count"]),
            "The compact expert-facing zip is regenerated and counted for handoff stability.",
        ),
    ]

    route_sync_rows = [
        row(
            "route_next_branch",
            "pass",
            "next official branch",
            1675.0,
            "The next official branch is `.1675-.1678` conditional reactivation.",
        ),
        row(
            "route_primary_reopen_surface",
            "pass",
            "primary reopen surface fixed",
            1.0,
            "A genuinely new action-level structure remains the primary reopen trigger.",
        ),
        row(
            "route_secondary_reopen_surface",
            "pass",
            "secondary reopen surface fixed",
            1.0,
            "Constitutive-map or branch-local full nonlinear energy-density reopening stays secondary after a pack update.",
        ),
        row(
            "route_reserve_reopen_surface",
            "pass",
            "reserve reopen surface fixed",
            1.0,
            "Future external input remains reserve and does not restart the current pack by itself.",
        ),
        row(
            "route_staging_file_count",
            "pass",
            "staging file count",
            float(bundle["staging_file_count"]),
            "The staged expert-facing pack includes copied sources plus summary text files.",
        ),
    ]

    evidence = {
        "retained_scalar_exact_alpha_at_q_theory": SCALAR_ALPHA,
        "official_energy_core_alpha_at_q_theory": ENERGY_ALPHA,
        "official_projected_kernel_alpha_at_q_theory": PROJECTED_ALPHA,
        "electric_like_component_alpha_at_q_theory": ELECTRIC_LIKE_ALPHA,
        "note_gradient_alpha_at_q_theory": NOTE_GRADIENT_ALPHA,
        "bundle_dir": bundle["bundle_dir"],
        "bundle_zip": bundle["bundle_zip"],
        "prior_registry_summary": prior_summary,
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "same_level_fallback_family_exhausted": True,
        "same_level_rescue_lane_extension_admissible": False,
        "noncanonical_scalar_leaning_evidence_retained": True,
        "exact_constitutive_map_available": False,
        "branch_local_full_nonlinear_energy_density_exact_available": False,
        "projected_kernel_fallback_failed": True,
        "constrained_ground_state_branch_selection_supported": False,
        "primary_reopen_surface": PRIMARY_REOPEN,
        "secondary_reopen_surface": SECONDARY_REOPEN,
        "reserve_reopen_surface": RESERVE_REOPEN,
        "fallback_closeout_advice_pack_refresh_completed": advice_pack_ready,
        "conditional_reactivation_only_after_new_surface_or_input": True,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": False,
        "retained_scalar_exact_alpha_at_q_theory": SCALAR_ALPHA,
        "official_energy_core_alpha_at_q_theory": ENERGY_ALPHA,
        "official_projected_kernel_alpha_at_q_theory": PROJECTED_ALPHA,
        "electric_like_component_alpha_at_q_theory": ELECTRIC_LIKE_ALPHA,
        "note_gradient_alpha_at_q_theory": NOTE_GRADIENT_ALPHA,
        "bundle_refresh_count": float(bundle["zip_file_count"]),
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": advice_pack_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    inventory_payload = payload(
        "8.7.56.1671",
        f"{STEP_NAME} inventory",
        inputs,
        inventory_rows,
        summary,
        decision,
        {
            "status_hit": hit(status_text, "fallback closeout advice-pack refresh"),
            "roadmap_hit": hit(roadmap_text, "8.7.56.1671-.1674"),
            "current_problem_hit": hit(current_problem_text, "fallback closeout advice-pack refresh"),
            "current_status_hit": hit(current_status_text, "fallback closeout advice-pack refresh"),
        },
    )
    audit_payload = payload(
        "8.7.56.1672",
        f"{STEP_NAME} audit",
        inputs,
        audit_rows,
        summary,
        decision,
        {
            "unified_roadmap_hit": hit(unified_text, ".1671-.1674"),
            "part5_hit": hit(part5_text, ".1671-.1674"),
            "local_response_hit": hit(local_response_text, "conditional new action-level structure"),
            "ai_context_next": ai_context["next"],
        },
    )
    declaration_payload = payload(
        "8.7.56.1673",
        f"{STEP_NAME} declaration gate",
        inputs,
        declaration_rows,
        summary,
        decision,
        evidence,
    )
    route_sync_payload = payload(
        "8.7.56.1674",
        f"{STEP_NAME} route sync",
        inputs,
        route_sync_rows,
        summary,
        decision,
        evidence,
    )

    outputs = {
        "inventory": write_artifact("inventory", inventory_payload),
        "audit": write_artifact("audit", audit_payload),
        "declaration_gate": write_artifact("declaration_gate", declaration_payload),
        "route_sync": write_artifact("route_sync", route_sync_payload),
    }

    print(
        json.dumps(
            {
                "branch_class": BRANCH_CLASS,
                "next_official_branch": NEXT_BRANCH,
                "selected_next_generation_route": NEXT_ROUTE_NAME,
                "recommended_next_route_or_none": NEXT_ROUTE,
                "bundle_dir": bundle["bundle_dir"],
                "bundle_zip": bundle["bundle_zip"],
                "copied_count": bundle["copied_count"],
                "staging_file_count": bundle["staging_file_count"],
                "zip_file_count": bundle["zip_file_count"],
                "outputs": outputs,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
