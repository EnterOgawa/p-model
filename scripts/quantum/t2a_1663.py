#!/usr/bin/env python3
"""Generate 8.7.56.1663-.1666 constrained ground-state fallback artifacts.

This branch is the second fallback after the projected-kernel transverse
response also failed honestly on the retained vector no-go scale.

The question is intentionally narrow:

Can the current frozen-action pack already justify a branch switch via one
constrained ground-state / branch-selection theorem?

The current pack would need all of the following to answer "yes":

1. the projected-kernel fallback would need to leave an unresolved response
   loophole rather than closing negatively,
2. the retained exact pilot family would need a supported nodeless /
   ground-state branch-selection story,
3. the theorem-side two-component `ell=0` closure would need to already be
   implied under the current public + solver pack.

What we actually have is the opposite:

- the projected-kernel observable also lands on the retained vector no-go
  scale,
- the prior nodeless/ground-state note failed under the current exact pilot,
- the older theorem-side review already fixed that explicit `ell=0`
  two-component closure is not implied under the current pack.

This script therefore freezes the honest result:

    constrained ground-state / branch-selection is not supported

and sends the route to fallback closeout / reopen registry.
"""

from __future__ import annotations

import csv
import json
import math
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

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_ground_state_identification_20260328.md"
)
PRIOR_FALLBACK_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1659_1662_pmu_tresp_pk_audit_declaration_gate_metrics.json"
)
GROUND_STATE_NOTE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1615_1618_gs_nodeless_audit_declaration_gate_metrics.json"
)
ANCHOR_CONTINUATION_EVAL = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
)
GROUND_STATE_CLOSURE_REVIEW = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_"
    "ground_state_two_component_closure_review_declaration_gate_metrics.json"
)
GROUND_STATE_CLOSURE_CONTRACT = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_"
    "ground_state_two_component_closure_gap_contract_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1663-1666"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor constrained ground-state / "
    "branch-selection audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "gs_branch_select_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_p_mu_transverse_response_projected_kernel_"
    "tracks_vector_no_go_ground_state_fallback_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_constrained_ground_state_branch_selection_"
    "not_supported_fallback_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_closeout_reopen_registry"
)
NEXT_ROUTE = "8.7.56.1667"

SCALAR_ALPHA = 0.00715678583937324
PROJECTED_KERNEL_ALPHA = 0.0005600186431488893
TARGET_ALPHA = 1.0 / 137.035999084


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

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


# 関数: 表示用の相対パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を構成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を構成する。

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


# 関数: JSON/CSV 成果物を書き出す。

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


# 関数: target 相対残差を返す。

def alpha_residual_rel(alpha_value: float) -> float:
    """Return one target-relative residual."""
    return float(abs(float(alpha_value) - TARGET_ALPHA) / TARGET_ALPHA)


# 関数: `.1663-.1666` を実行する。

def main() -> None:
    """Execute the constrained ground-state / branch-selection fallback audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        PRIOR_FALLBACK_GATE,
        GROUND_STATE_NOTE_GATE,
        ANCHOR_CONTINUATION_EVAL,
        GROUND_STATE_CLOSURE_REVIEW,
        GROUND_STATE_CLOSURE_CONTRACT,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)
    directive_text = read_text(DIRECTIVE_NOTE) if DIRECTIVE_NOTE.exists() else ""

    prior_fallback_summary = read_json(PRIOR_FALLBACK_GATE)["summary"]
    ground_state_note_summary = read_json(GROUND_STATE_NOTE_GATE)["summary"]
    anchor_eval_summary = read_json(ANCHOR_CONTINUATION_EVAL)["summary"]
    closure_review_summary = read_json(GROUND_STATE_CLOSURE_REVIEW)["summary"]
    closure_contract_summary = read_json(GROUND_STATE_CLOSURE_CONTRACT)["summary"]

    prior_transverse_fallback_failed = bool(
        prior_fallback_summary.get("trial2_numeric_alpha_problem_classification")
        == PRIOR_CLASS
        and prior_fallback_summary.get("transverse_response_fallback_failed", False)
    )
    prior_nodeless_hypothesis_supported = bool(
        ground_state_note_summary.get(
            "ground_state_nodeless_hypothesis_supported_under_current_pack", False
        )
    )
    prior_ground_state_closure_implied = bool(
        closure_review_summary.get(
            "ground_state_two_component_closure_already_implied_under_current_pack",
            False,
        )
        or closure_contract_summary.get(
            "ground_state_two_component_closure_already_implied_under_current_pack",
            False,
        )
    )
    explicit_ell0_ground_state_two_component_closure_available = bool(
        closure_review_summary.get(
            "explicit_ell0_ground_state_two_component_closure_available", False
        )
    )
    phase1_equivalent_row = anchor_eval_summary["phase1_equivalent_row"]
    phase1_branch_noded = bool(
        phase1_equivalent_row["node_count_k0"] > 0
        or phase1_equivalent_row["node_count_kL"] > 0
    )

    projected_kernel_residual_rel = float(
        prior_fallback_summary["official_projected_kernel_alpha_residual_rel"]
    )
    scalar_residual_rel = alpha_residual_rel(SCALAR_ALPHA)
    residual_ratio_vs_scalar = float(projected_kernel_residual_rel / scalar_residual_rel)
    scalar_candidate_retained_but_noncanonical = bool(SCALAR_ALPHA < TARGET_ALPHA)

    constrained_ground_state_branch_selection_supported = bool(
        prior_transverse_fallback_failed
        and prior_nodeless_hypothesis_supported
        and prior_ground_state_closure_implied
        and explicit_ell0_ground_state_two_component_closure_available
        and not phase1_branch_noded
    )
    fallback_family_exhausted = bool(
        prior_transverse_fallback_failed
        and not prior_nodeless_hypothesis_supported
        and not prior_ground_state_closure_implied
        and phase1_branch_noded
    )
    fallback_closeout_reopen_registry_required = bool(
        fallback_family_exhausted and not constrained_ground_state_branch_selection_supported
    )
    physical_reject_required = False

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "constrained ground-state / branch-selection"),
            hit(roadmap_text, "8.7.56.1663-.1666"),
            hit(current_problem_text, "constrained ground-state / branch-selection"),
            hit(current_status_text, "constrained ground-state / branch-selection"),
            hit(
                unified_text,
                "`.1663-.1666` は **constrained ground-state / branch-selection audit**",
            ),
            hit(part5_text, "constrained ground-state / branch-selection audit"),
        )
    )
    constrained_branch_selection_wording_honest = bool(
        inventory_ready
        and prior_transverse_fallback_failed
        and not prior_nodeless_hypothesis_supported
        and not prior_ground_state_closure_implied
        and not explicit_ell0_ground_state_two_component_closure_available
        and phase1_branch_noded
        and not constrained_ground_state_branch_selection_supported
        and fallback_closeout_reopen_registry_required
        and not physical_reject_required
    )
    route_sync_ready = bool(constrained_branch_selection_wording_honest)

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "branch-selection inventory ready",
            truth(inventory_ready),
            "The second fallback only starts after the projected-kernel failure, prior nodeless note, older two-component closure gap, and roadmap wording all point to the same `.1663-.1666` branch.",
        ),
        row(
            "prior_transverse_fallback_failed",
            "pass" if prior_transverse_fallback_failed else "reject",
            "prior transverse-response fallback failed",
            truth(prior_transverse_fallback_failed),
            "A constrained branch-selection fallback is only admissible after the projected-kernel response observable has also failed honestly on the retained vector no-go scale.",
        ),
        row(
            "prior_nodeless_hypothesis_supported",
            "pass" if prior_nodeless_hypothesis_supported else "reject",
            "prior ground-state nodeless hypothesis supported",
            truth(prior_nodeless_hypothesis_supported),
            "The earlier ground-state note would need to open a nodeless branch-family story before a branch switch could be justified under the current pack.",
        ),
        row(
            "prior_ground_state_closure_implied",
            "pass" if prior_ground_state_closure_implied else "reject",
            "ground-state two-component closure already implied",
            truth(prior_ground_state_closure_implied),
            "The theorem-side pack would need to already imply the electron-like `ell=0` two-component closure before branch selection could be promoted honestly.",
        ),
        row(
            "explicit_ell0_ground_state_two_component_closure_available",
            "pass" if explicit_ell0_ground_state_two_component_closure_available else "reject",
            "explicit ell=0 ground-state two-component closure available",
            truth(explicit_ell0_ground_state_two_component_closure_available),
            "The current public + solver pack still does not expose an explicit `ell=0` two-component closure theorem for the retained exact vector branch.",
        ),
        row(
            "phase1_branch_noded",
            "watch" if phase1_branch_noded else "pass",
            "Phase-1-equivalent branch remains noded",
            truth(phase1_branch_noded),
            "The retained anchor-preserving branch still carries nonzero node counts, so there is no current-pack evidence that the fallback branch is already the constrained ground state.",
        ),
        row(
            "phase1_equivalent_node_total",
            "watch",
            "Phase-1-equivalent total node count",
            float(
                phase1_equivalent_row["node_count_k0"]
                + phase1_equivalent_row["node_count_kL"]
            ),
            "This combines the retained exact branch node counts that still survive at the Phase-1-equivalent point.",
        ),
        row(
            "phase1_equivalent_zero_radius",
            "watch",
            "Phase-1-equivalent first zero radius",
            float(ground_state_note_summary["phase1_zero_radius"]),
            "The retained exact pilot still crosses zero at finite radius, so the current fallback cannot claim a nodeless constrained-ground-state branch.",
        ),
        row(
            "projected_kernel_residual_rel",
            "reject",
            "projected-kernel residual relative to target",
            projected_kernel_residual_rel,
            "The first fallback already failed at the observable level and therefore cannot be reused as positive evidence for branch switching.",
        ),
        row(
            "projected_residual_ratio_vs_scalar",
            "watch",
            "projected-kernel residual divided by retained scalar residual",
            residual_ratio_vs_scalar,
            "This quantifies how much worse the projected-kernel fallback is than the retained scalar strong candidate under the same target residual measure.",
        ),
        row(
            "scalar_candidate_retained_but_noncanonical",
            "pass" if scalar_candidate_retained_but_noncanonical else "reject",
            "scalar candidate retained but noncanonical",
            truth(scalar_candidate_retained_but_noncanonical),
            "The scalar strong candidate still exists numerically, but the current branch-selection audit is about whether the present pack canonically licenses a branch switch. It does not.",
        ),
        row(
            "constrained_ground_state_branch_selection_supported",
            "pass" if constrained_ground_state_branch_selection_supported else "reject",
            "constrained ground-state / branch-selection supported",
            truth(constrained_ground_state_branch_selection_supported),
            "A positive result would require both theorem-side closure and branch-side ground-state support. The current pack satisfies neither.",
        ),
        row(
            "fallback_family_exhausted",
            "pass" if fallback_family_exhausted else "reject",
            "current fallback family exhausted",
            truth(fallback_family_exhausted),
            "Density, constitutive-map, nonlinear-energy, projected-kernel, and branch-selection fallback families have now all closed negatively under the current pack.",
        ),
        row(
            "fallback_closeout_reopen_registry_required",
            "pass" if fallback_closeout_reopen_registry_required else "reject",
            "fallback closeout / reopen registry required",
            truth(fallback_closeout_reopen_registry_required),
            "Once the second fallback also fails honestly, the correct next step is to freeze the current-pack result and its reopen surfaces rather than invent another rescue lane at the same level.",
        ),
        row(
            "physical_reject_required",
            "reject",
            "physical reject required",
            truth(physical_reject_required),
            "The failure remains route-local to the current pack. It does not force physical rejection of the broader scalar-side candidate structure.",
        ),
        row(
            "constrained_branch_selection_wording_honest",
            "pass" if constrained_branch_selection_wording_honest else "reject",
            "branch-selection wording honest",
            truth(constrained_branch_selection_wording_honest),
            "The wording is honest only if the prior projected fallback failed, the nodeless note stayed unsupported, the theorem-side closure remained absent, and the route now moves to closeout rather than to another same-level rescue.",
        ),
        row(
            "route_sync_ready",
            "pass" if route_sync_ready else "reject",
            "route sync ready",
            truth(route_sync_ready),
            "Once the second fallback is fixed negatively, the roadmap can move cleanly to fallback closeout / reopen registry.",
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
            "directive_note": display_path(DIRECTIVE_NOTE),
            "prior_fallback_gate": display_path(PRIOR_FALLBACK_GATE),
            "ground_state_note_gate": display_path(GROUND_STATE_NOTE_GATE),
            "anchor_continuation_eval": display_path(ANCHOR_CONTINUATION_EVAL),
            "ground_state_closure_review": display_path(GROUND_STATE_CLOSURE_REVIEW),
            "ground_state_closure_contract": display_path(
                GROUND_STATE_CLOSURE_CONTRACT
            ),
        },
        "constants": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "projected_kernel_alpha_at_q_theory": PROJECTED_KERNEL_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "prior_transverse_response_fallback_failed": prior_transverse_fallback_failed,
        "ground_state_nodeless_hypothesis_supported_under_current_pack": (
            prior_nodeless_hypothesis_supported
        ),
        "ground_state_two_component_closure_already_implied_under_current_pack": (
            prior_ground_state_closure_implied
        ),
        "explicit_ell0_ground_state_two_component_closure_available": (
            explicit_ell0_ground_state_two_component_closure_available
        ),
        "phase1_equivalent_node_count_k0": phase1_equivalent_row["node_count_k0"],
        "phase1_equivalent_node_count_kL": phase1_equivalent_row["node_count_kL"],
        "phase1_equivalent_zero_radius": ground_state_note_summary["phase1_zero_radius"],
        "phase1_equivalent_max_abs_ratio": phase1_equivalent_row["max_abs_ratio"],
        "projected_kernel_alpha_at_q_theory": prior_fallback_summary[
            "official_projected_kernel_alpha_at_q_theory"
        ],
        "projected_kernel_alpha_residual_rel": projected_kernel_residual_rel,
        "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
        "scalar_alpha_residual_rel": scalar_residual_rel,
        "projected_residual_ratio_vs_scalar": residual_ratio_vs_scalar,
        "scalar_candidate_retained_but_noncanonical": (
            scalar_candidate_retained_but_noncanonical
        ),
        "constrained_ground_state_branch_selection_supported": (
            constrained_ground_state_branch_selection_supported
        ),
        "fallback_family_exhausted": fallback_family_exhausted,
        "fallback_closeout_reopen_registry_required": (
            fallback_closeout_reopen_registry_required
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": route_sync_ready,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "hits": {
            "status_branch_hit": hit(
                status_text, "constrained ground-state / branch-selection"
            ),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1663-.1666"),
            "current_problem_branch_hit": hit(
                current_problem_text, "constrained ground-state / branch-selection"
            ),
            "current_status_branch_hit": hit(
                current_status_text, "constrained ground-state / branch-selection"
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1663-.1666` は **constrained ground-state / branch-selection audit**",
            ),
            "part5_branch_hit": hit(
                part5_text, "constrained ground-state / branch-selection audit"
            ),
            "directive_ground_state_hit": hit(
                directive_text, "ground state の条件"
            ),
            "directive_nodeless_hit": hit(directive_text, "nodeless"),
        },
        "carry_over": {
            "prior_fallback_summary": prior_fallback_summary,
            "ground_state_note_summary": ground_state_note_summary,
            "anchor_eval_summary": anchor_eval_summary,
            "closure_review_summary": closure_review_summary,
            "closure_contract_summary": closure_contract_summary,
        },
        "retained_numeric_state": {
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "projected_kernel_alpha_at_q_theory": PROJECTED_KERNEL_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1663",
                f"{STEP_NAME} inventory",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "audit": write_artifact(
            "audit",
            payload(
                "8.7.56.1664",
                f"{STEP_NAME} audit",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1665",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload(
                "8.7.56.1666",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "manifest": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
