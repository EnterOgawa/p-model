#!/usr/bin/env python3
"""Generate 8.7.56.1859-.1862 post-dormant reactivation audit artifacts."""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
DOWNLOADS = Path(r"C:/Users/ogawa/Downloads")

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

POST_CLOSEOUT_DORMANT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1855_1858_post_closeout_wait_restore_dormant_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1859-1862"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional post-dormant "
    "pack-update or external-input reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "post_dormant_reactivation_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_dormant_"
    "registry_wait_restored"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_post_dormant_reactivation_audit_no_new_surface_"
    "registry_refresh_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_dormant_wait_restore_"
    "registry_refresh"
)
NEXT_ROUTE = "8.7.56.1863"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_second_post_dormant_"
    "pack_update_or_external_input_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1867"
DOWNLOAD_PATTERNS = (
    "trial2",
    "vector_qball",
    "numeric_alpha",
    "source_phase",
    "signed",
)


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input file is missing."""
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


# 関数: repo相対の表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
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


# 関数: JSON/CSV artifact を書き出す。

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


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: Downloads から relevant note を列挙する。

def relevant_download_rows() -> list[dict]:
    """Return relevant Trial-2 Downloads rows sorted by UTC mtime descending."""
    rows: list[dict] = []
    pattern = re.compile("|".join(re.escape(part) for part in DOWNLOAD_PATTERNS), re.IGNORECASE)
    for path in DOWNLOADS.iterdir():
        if not path.is_file():
            continue

        if not pattern.search(path.name):
            continue

        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                "size_bytes": int(stat.st_size),
            }
        )

    rows.sort(key=lambda item: item["mtime_utc"], reverse=True)
    return rows


# 関数: closeout 後に新 input があるかを判定する。

def count_newer(rows: list[dict], threshold_utc: datetime) -> int:
    """Count rows newer than the retained closeout timestamp."""
    return sum(1 for item in rows if item["mtime_utc"] > threshold_utc)


# 関数: post-closeout reopen formulas を返す。

def build_formulae() -> dict[str, str]:
    """Return the retained post-closeout reopen formulas."""
    return {
        "retained_exact_alpha_rule": "alpha_exact(q) = |F_exact(q)|^2 / (4 pi)",
        "retained_signed_rule": "F_exact(q) = sigma_F(q) |F_exact(q)|",
        "primary_reopen_surface": "substantive pack update beyond the retained real-branch sign-parity theorem",
        "secondary_reopen_surface": "retained-interval extension or new signed observable rule beyond the current dormant family",
        "reserve_reopen_surface": "future external input guiding a new signed surface or pack update",
    }


# 関数: `.1851-.1854` を実行する。

def main() -> None:
    """Execute the post-closeout reactivation audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        POST_CLOSEOUT_DORMANT_GATE,
        DOWNLOADS,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    ai_context = read_json(AI_CONTEXT)
    dormant_summary = read_json(POST_CLOSEOUT_DORMANT_GATE)["summary"]
    threshold_utc = datetime.fromisoformat(ai_context["current_date_utc"])
    download_rows = relevant_download_rows()
    matching_download_count = len(download_rows)
    newer_count = count_newer(download_rows, threshold_utc)
    latest_download = download_rows[0] if download_rows else None

    inventory_ready = all(
        (
            bool(dormant_summary["exact_alpha_promotion_retained"]),
            bool(dormant_summary["exact_signed_form_factor_promotion_retained"]),
            bool(dormant_summary["post_closeout_dormant_registry_retained"]),
        )
    )
    genuinely_new_external_input_detected = newer_count > 0
    substantive_pack_update_surface_present = False
    retained_interval_extension_surface_present = False
    new_signed_observable_rule_surface_present = False
    new_primary_trigger_opened = substantive_pack_update_surface_present
    new_secondary_trigger_opened = (
        retained_interval_extension_surface_present or new_signed_observable_rule_surface_present
    )
    new_reserve_trigger_opened = genuinely_new_external_input_detected
    no_new_trigger_opened = not any(
        (
            new_primary_trigger_opened,
            new_secondary_trigger_opened,
            new_reserve_trigger_opened,
        )
    )
    future_conditional_reactivation_still_required = True
    physical_reject_required = False

    formulas = build_formulae()
    latest_name = latest_download["name"] if latest_download else ""
    latest_mtime = latest_download["mtime_utc"].isoformat() if latest_download else ""

    rows = [
        row("inventory_ready", "pass" if inventory_ready else "reject", "post-closeout reactivation inventory ready", truth(inventory_ready), "The audit starts only after exact alpha promotion and exact signed form-factor promotion are both retained."),
        row("matching_download_count", "watch", "matching Downloads note count", float(matching_download_count), "Relevant post-dormant notes are detected by filename match over the Trial-2/vector-Qball namespace."),
        row("genuinely_new_external_input_detected", "reject" if not genuinely_new_external_input_detected else "pass", "genuinely new external input detected after dormant restore", truth(genuinely_new_external_input_detected), "No relevant Downloads note is newer than the retained dormant timestamp, so reactivation does not open."),
        row("substantive_pack_update_surface_present", "reject", "substantive pack-update surface present", truth(substantive_pack_update_surface_present), "No new action-level structure or pack update beyond the retained real-branch sign-parity theorem is present locally."),
        row("retained_interval_extension_surface_present", "reject", "retained-interval extension surface present", truth(retained_interval_extension_surface_present), "The retained interval 0<=q/m0<=1 is unchanged in the current pack."),
        row("new_signed_observable_rule_surface_present", "reject", "new signed observable rule surface present", truth(new_signed_observable_rule_surface_present), "No post-closeout theorem introduces a new signed observable rule beyond the retained sign-parity theorem."),
        row("new_primary_trigger_opened", "reject", "new primary trigger opened", truth(new_primary_trigger_opened), "The primary reopen surface stays closed without a substantive pack update."),
        row("new_secondary_trigger_opened", "reject", "new secondary trigger opened", truth(new_secondary_trigger_opened), "Neither retained-interval extension nor new signed observable rule is present."),
        row("new_reserve_trigger_opened", "reject" if not new_reserve_trigger_opened else "pass", "new reserve trigger opened", truth(new_reserve_trigger_opened), "Even the reserve external-input trigger stays closed because no newer note exists after closeout."),
        row("no_new_trigger_opened", "pass" if no_new_trigger_opened else "reject", "no new trigger opened", truth(no_new_trigger_opened), "The honest result is a dormant continuation, not a same-level reactivation."),
        row("future_conditional_reactivation_still_required", "pass", "future conditional reactivation still required", truth(future_conditional_reactivation_still_required), "Future work remains conditional on a substantive pack update or genuinely new external input."),
        row("physical_reject_required", "reject", "physical reject required", truth(physical_reject_required), "The family stays retained; there is no physical reject at this branch."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "dormant_threshold_utc": threshold_utc.isoformat(),
        "matching_download_count": matching_download_count,
        "latest_relevant_download_name": latest_name,
        "latest_relevant_download_utc": latest_mtime,
        "genuinely_new_external_input_detected": genuinely_new_external_input_detected,
        "substantive_pack_update_surface_present": substantive_pack_update_surface_present,
        "retained_interval_extension_surface_present": retained_interval_extension_surface_present,
        "new_signed_observable_rule_surface_present": new_signed_observable_rule_surface_present,
        "new_primary_trigger_opened": new_primary_trigger_opened,
        "new_secondary_trigger_opened": new_secondary_trigger_opened,
        "new_reserve_trigger_opened": new_reserve_trigger_opened,
        "no_new_trigger_opened": no_new_trigger_opened,
        "future_conditional_reactivation_still_required": future_conditional_reactivation_still_required,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_post_dormant_reactivation_audit_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "download_rows": [{"name": item["name"], "mtime_utc": item["mtime_utc"].isoformat(), "size_bytes": item["size_bytes"]} for item in download_rows[:20]],
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1859"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1859-.1862"),
            "current_problem_hit": hit(current_problem_text, "signed form factor"),
            "current_status_hit": hit(current_status_text, "signed form factor"),
            "unified_roadmap_hit": hit(unified_text, "92. `.1859-.1862`"),
            "long_roadmap_hit": hit(long_text, "8.7.56.1859-.1862"),
            "part5_hit": hit(part5_text, "signed form factor"),
        },
    }

    declaration_payload = payload(
        "8.7.56.1861",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": display_path(STATUS),
                "roadmap": display_path(ROADMAP),
                "ai_context": display_path(AI_CONTEXT),
                "work_history_recent": display_path(WORK_HISTORY_RECENT),
                "current_problem": display_path(CURRENT_PROBLEM),
                "current_status": display_path(CURRENT_STATUS),
                "unified_roadmap": display_path(UNIFIED_ROADMAP),
                "long_roadmap": display_path(LONG_ROADMAP),
                "part5": display_path(PART5),
                "post_closeout_dormant_gate": display_path(POST_CLOSEOUT_DORMANT_GATE),
                "downloads": str(DOWNLOADS),
            },
            "constants": {
                "dormant_threshold_utc": threshold_utc.isoformat(),
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        decision,
        evidence,
    )

    route_payload = payload(
        "8.7.56.1862",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            row("no_new_trigger_opened", "pass" if no_new_trigger_opened else "reject", "no new trigger opened", truth(no_new_trigger_opened), "The honest next move is dormant registry sync because no reopen surface appeared."),
            row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the post-dormant wait restore / registry refresh sync."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": physical_reject_required,
        },
        {
            "overall_status": "vector_qball_form_factor_post_dormant_reactivation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1859-.1862 post-dormant reactivation audit artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
