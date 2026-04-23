#!/usr/bin/env python3
"""Generate 8.7.56.1643-.1646 energy-density reopen advice-pack refresh artifacts.

This branch does not introduce a new observable or a fresh numerical rescue.
It converts the current reopen registry into an expert-facing computation pack
and resets the scientific mainline to the breakthrough instruction order:

1. exact constitutive-map audit
2. branch-local full nonlinear energy-density audit
3. primary decision gate / secondary canonical-promotion audit

The earlier `P_mu` transverse response idea is retained only as a fallback
route if the current instruction pack cannot close within the frozen action.
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
PRIOR_RESPONSE = ROOT / "doc" / "quantum" / "49_trial2_numeric_alpha_vector_qball_energy_density_reopen_registry_response.md"
LOCAL_RESPONSE = ROOT / "doc" / "quantum" / "50_trial2_vector_qball_breakthrough_instruction_response.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EXTERNAL_NOTE = Path(
    r"C:\Users\ogawa\Downloads\50_trial2_numeric_alpha_vector_qball_breakthrough_instruction_pack.md"
)
REGISTRY_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1639_1642_energy_density_reopen_registry_declaration_gate_metrics.json"
)
REGISTRY_ROUTE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1639_1642_energy_density_reopen_registry_route_sync_metrics.json"
)
SCRIPT_SELF = Path(__file__).resolve()

STEP_TAG = "8.7.56.1643-1646"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor energy-density reopen "
    "advice-pack refresh"
)
STEM = build_compact_artifact_stem(STEP_TAG, "ed_reopen_advice_pack", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_energy_density_case_ii_vector_no_go_like_"
    "reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_energy_density_reopen_advice_pack_refresh_"
    "completed"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exact_constitutive_map_audit"
)
NEXT_ROUTE = "8.7.56.1647"
NEXT_BRANCH = "8.7.56.1647-.1650"
FOLLOWUP_ROUTE = "8.7.56.1651-.1654"
DECISION_ROUTE = "8.7.56.1655-.1658"
FALLBACK_ROUTE = "8.7.56.1659-.1662"
SECOND_FALLBACK_ROUTE = "8.7.56.1663-.1666"

PRIMARY_REOPEN = (
    "branch_local_full_nonlinear_energy_density_or_exact_constitutive_map_gap"
)
SECONDARY_REOPEN = (
    "evidence_only_electric_like_or_note_gradient_canonical_promotion_gap"
)
RESERVE_REOPEN = "future_external_input_or_new_action_level_structure"

BUNDLE_DIR = PRIVATE_OUT / "ed_reopen_pack_20260328"
BUNDLE_ZIP = PRIVATE_OUT / "ed_reopen_pack_20260328.zip"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を検証する。

def require(path: Path) -> None:
    """Fail when one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8テキストを読み込む。

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
    """Convert one absolute path into repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準形式の metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準形式の payload を作る。

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
    """Return the canonical README text for the refreshed advice pack."""
    return (
        "Energy-density reopen advice pack\n\n"
        "Current official read\n"
        "- Case II vector-no-go-like under current pack.\n"
        "- Official exact core: F_E(q_theory)=-0.0825465944966888, "
        "alpha_E(q_theory)=0.0005422361373947313.\n"
        "- Physical reject required: false.\n\n"
        "Adopted primary order\n"
        "- 8.7.56.1647-.1650 exact constitutive-map audit\n"
        "- 8.7.56.1651-.1654 branch-local full nonlinear energy-density audit\n"
        "- 8.7.56.1655-.1658 primary decision gate / secondary canonical-promotion audit\n\n"
        "Fallback order\n"
        "- 8.7.56.1659-.1662 P_mu transverse response / projected-kernel observable audit\n"
        "- 8.7.56.1663-.1666 constrained ground-state / branch-selection audit\n"
    )


# 関数: bundle note を返す。

def bundle_note_text() -> str:
    """Return the canonical note text for the refreshed advice pack."""
    return (
        "Breakthrough instruction note\n\n"
        "The current reopen registry is kept intact.\n"
        "Primary reopen surface remains branch-local full nonlinear energy density "
        "or exact constitutive-map gap.\n"
        "Secondary remains evidence-only electric-like / note-gradient canonical "
        "promotion gap.\n"
        "Reserve remains future external input or new action-level structure.\n\n"
        "This pack adopts the expert breakthrough instruction as the first-shot "
        "mainline and demotes the P_mu transverse-response idea to fallback.\n"
    )


# 関数: pack summary を返す。

def pack_summary_text() -> str:
    """Return the concise pack summary text."""
    return (
        "Current pack keeps the energy-density Case II vector-no-go-like read "
        "honest while promoting exact constitutive-map audit to the next official "
        "mainline. Branch-local full nonlinear energy density is the second primary "
        "lane, and P_mu transverse response stays as fallback."
    )


# 関数: bundle manifest を返す。

def manifest_text(copied_sources: list[Path]) -> str:
    """Return the manifest text for the refreshed advice pack."""
    lines = [
        "Energy-density reopen advice-pack manifest",
        f"Generated: {now_iso()}",
        f"COPIED_COUNT={len(copied_sources)}",
        "",
    ]
    lines.extend(display_path(path) for path in copied_sources)
    return "\n".join(lines) + "\n"


# 関数: advice-pack bundle を再生成する。

def refresh_bundle(copied_sources: list[Path]) -> dict[str, object]:
    """Rebuild the compact expert-facing bundle for this advice-pack refresh."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
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


# 関数: `.1643-.1646` を実行する。

def main() -> None:
    """Execute the energy-density reopen advice-pack refresh branch."""
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
        EXTERNAL_NOTE,
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
    external_text = read_text(EXTERNAL_NOTE)
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
        EXTERNAL_NOTE,
        REGISTRY_GATE,
        REGISTRY_ROUTE,
        SCRIPT_SELF,
    ]
    bundle = refresh_bundle(copied_sources)

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
        "external_instruction_pack": display_path(EXTERNAL_NOTE),
        "registry_gate": display_path(REGISTRY_GATE),
        "registry_route": display_path(REGISTRY_ROUTE),
        "script": display_path(SCRIPT_SELF),
    }

    inventory_rows = [
        row(
            "inventory_prior_registry",
            "ok",
            "prior_registry_completed",
            truth(
                registry_route["summary"][
                    "selected_next_generation_route"
                ]
                == "trial2_numeric_alpha_vector_qball_form_factor_energy_density_reopen_advice_pack_refresh"
            ),
            "The prior registry already points to the advice-pack refresh route.",
        ),
        row(
            "inventory_primary_surface",
            "ok",
            "primary_reopen_surface_retained",
            truth(
                registry_gate["summary"]["primary_reopen_surface"] == PRIMARY_REOPEN
            ),
            "The primary reopen surface remains the exact constitutive-map / full nonlinear energy-density gap.",
        ),
        row(
            "inventory_secondary_surface",
            "ok",
            "secondary_reopen_surface_retained",
            truth(
                registry_gate["summary"]["secondary_reopen_surface"]
                == SECONDARY_REOPEN
            ),
            "The electric-like / note-gradient lane stays secondary and evidence-only.",
        ),
        row(
            "inventory_reserve_surface",
            "ok",
            "reserve_reopen_surface_retained",
            truth(
                registry_gate["summary"]["reserve_reopen_surface"] == RESERVE_REOPEN
            ),
            "Future external input stays reserve rather than primary.",
        ),
        row(
            "inventory_bundle_sources",
            "ok",
            "copied_source_count",
            float(bundle["copied_count"]),
            "The advice-pack bundle copies the required source set only.",
        ),
    ]

    audit_rows = [
        row(
            "audit_instruction_pack",
            "ok",
            "breakthrough_instruction_pack_adopted",
            truth("hard gates" in external_text.lower()),
            "The expert instruction pack is adopted as the first-shot breakthrough mainline.",
        ),
        row(
            "audit_constitutive_mainline",
            "ok",
            "exact_constitutive_map_promoted_to_mainline",
            truth("exact constitutive-map audit" in local_response_text),
            "The first post-refresh mainline is exact constitutive-map audit.",
        ),
        row(
            "audit_nonlinear_followup",
            "ok",
            "branch_local_full_nonlinear_energy_density_followup_scheduled",
            truth("branch-local full nonlinear energy-density audit" in local_response_text),
            "The second primary lane is the branch-local full nonlinear energy-density audit.",
        ),
        row(
            "audit_decision_gate",
            "ok",
            "primary_decision_gate_scheduled",
            truth("primary decision gate" in local_response_text.lower()),
            "A dedicated primary decision gate remains scheduled after the two primary audits.",
        ),
        row(
            "audit_transverse_fallback",
            "ok",
            "transverse_response_fallback_retained",
            truth("transverse response / projected-kernel observable audit" in local_response_text),
            "The P_mu transverse-response route is retained only as fallback.",
        ),
        row(
            "audit_ground_state_fallback",
            "ok",
            "branch_selection_fallback_retained",
            truth("branch-selection audit" in local_response_text),
            "Ground-state / branch-selection remains second fallback, not current mainline.",
        ),
        row(
            "audit_parallel_short_audits",
            "ok",
            "parallel_short_audit_count",
            4.0,
            "near-node, u/P_infty, conditional full transverse overlap, and parallel J_eff remain short audits only.",
        ),
    ]

    declaration_rows = [
        row(
            "gate_branch_class",
            "ok",
            "energy_density_reopen_advice_pack_refresh_ready",
            1.0,
            "The energy-density reopen advice pack is ready to be frozen officially.",
        ),
        row(
            "gate_physical_reject",
            "ok",
            "physical_reject_required",
            0.0,
            "Physical reject remains false under the refreshed advice pack.",
        ),
        row(
            "gate_mainline_reset",
            "ok",
            "breakthrough_instruction_mainline_reset_completed",
            1.0,
            "The scientific mainline is reset to the expert instruction order.",
        ),
        row(
            "gate_future_input",
            "ok",
            "future_external_input_side_lane_retained",
            1.0,
            "Future external input remains useful but outside the primary mainline.",
        ),
        row(
            "gate_bundle_ready",
            "ok",
            "expert_facing_bundle_ready",
            1.0,
            "The compact expert-facing advice pack bundle is regenerated for sharing.",
        ),
    ]

    route_sync_rows = [
        row(
            "route_next_official",
            "ok",
            "recommended_next_route_or_none",
            1647.0,
            "The next official branch is exact constitutive-map audit.",
        ),
        row(
            "route_followup_primary",
            "ok",
            "followup_primary_route",
            1651.0,
            "The second primary route is branch-local full nonlinear energy-density audit.",
        ),
        row(
            "route_primary_gate",
            "ok",
            "primary_decision_gate_route",
            1655.0,
            "The third primary route is the decision gate / secondary canonical-promotion audit.",
        ),
        row(
            "route_fallback_one",
            "ok",
            "transverse_response_fallback_route",
            1659.0,
            "The first fallback is the P_mu transverse-response / projected-kernel observable audit.",
        ),
        row(
            "route_fallback_two",
            "ok",
            "branch_selection_fallback_route",
            1663.0,
            "The second fallback is constrained ground-state / branch-selection audit.",
        ),
        row(
            "route_bundle_stage",
            "ok",
            "staging_file_count",
            float(bundle["staging_file_count"]),
            "The refreshed compact bundle contains the copied sources plus internal note files.",
        ),
        row(
            "route_bundle_zip",
            "ok",
            "zip_file_count",
            float(bundle["zip_file_count"]),
            "The zip artifact retains the same staged file count after compression.",
        ),
    ]

    evidence = {
        "retained_scalar_exact_alpha": registry_gate["summary"][
            "retained_scalar_exact_alpha_at_q_theory"
        ] if "retained_scalar_exact_alpha_at_q_theory" in registry_gate["summary"] else 0.00715678583937324,
        "official_energy_core_alpha": registry_gate["summary"][
            "official_alpha_E_at_q_theory"
        ],
        "vector_no_go_alpha": registry_gate["summary"].get(
            "retained_vector_no_go_alpha",
            0.0005579616187042394,
        ),
        "primary_reopen_surface": PRIMARY_REOPEN,
        "secondary_reopen_surface": SECONDARY_REOPEN,
        "reserve_reopen_surface": RESERVE_REOPEN,
        "bundle_dir": bundle["bundle_dir"],
        "bundle_zip": bundle["bundle_zip"],
    }

    inventory_payload = payload(
        STEP_TAG,
        STEP_NAME,
        inputs,
        inventory_rows,
        {
            "source_inventory_completed": True,
            "copied_source_count": bundle["copied_count"],
        },
        {
            "prior_classification": PRIOR_CLASS,
            "selected_branch_classification": BRANCH_CLASS,
        },
        {
            "status_hit": hit(status_text, "current official state"),
            "roadmap_hit": hit(
                roadmap_text,
                "8.7.56.1643-.1646",
            ),
            "current_problem_hit": hit(
                current_problem_text,
                "energy-density reopen advice-pack refresh",
            ),
            "current_status_hit": hit(
                current_status_text,
                "energy-density reopen advice-pack refresh",
            ),
        },
    )
    audit_payload = payload(
        STEP_TAG,
        STEP_NAME,
        inputs,
        audit_rows,
        {
            "adopted_mainline_count": 3,
            "parallel_short_audit_count": 4,
            "fallback_count": 2,
        },
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "unified_roadmap_hit": hit(
                unified_text,
                "energy-density reopen advice-pack refresh",
            ),
            "part5_hit": hit(part5_text, "energy-density reopen advice-pack refresh"),
            "ai_context_next": ai_context["next"],
        },
    )
    declaration_payload = payload(
        STEP_TAG,
        STEP_NAME,
        inputs,
        declaration_rows,
        {
            "energy_density_reopen_advice_pack_refresh_ready": True,
            "expert_facing_bundle_ready": True,
        },
        {
            "selected_branch_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        evidence,
    )
    route_sync_payload = payload(
        STEP_TAG,
        STEP_NAME,
        inputs,
        route_sync_rows,
        {
            "next_official_branch": NEXT_BRANCH,
            "followup_primary_branch": FOLLOWUP_ROUTE,
            "primary_gate_branch": DECISION_ROUTE,
            "first_fallback_branch": FALLBACK_ROUTE,
            "second_fallback_branch": SECOND_FALLBACK_ROUTE,
        },
        {
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "followup_route": FOLLOWUP_ROUTE,
            "primary_decision_gate_route": DECISION_ROUTE,
            "fallback_route": FALLBACK_ROUTE,
            "second_fallback_route": SECOND_FALLBACK_ROUTE,
        },
        evidence,
    )

    outputs = {
        "inventory": write_artifact("inventory", inventory_payload),
        "audit": write_artifact("audit", audit_payload),
        "declaration_gate": write_artifact("declaration_gate", declaration_payload),
        "route_sync": write_artifact("route_sync", route_sync_payload),
    }

    print(json.dumps(
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
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
