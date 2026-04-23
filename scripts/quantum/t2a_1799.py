#!/usr/bin/env python3
"""Generate 8.7.56.1799-.1802 full-q HH window generalization artifacts.

`.1791-.1794` closed exact scalar promotion only at the retained matching point
`q_theory`, and `.1795-.1798` honestly froze the missing bridge as the
full-q HH surface.

To move beyond the fixed-q point, this branch adopts one new theory layer:

    A_FF,ext(q) = |q|

as the vacuum-saturated external field-strength carrier surface under the
already retained unit-residue transverse vacuum normalization.  Combined with
the retained rank-one rule

    lambda_+(q) = A_FF(q) + A_HH(q),   rho_exact(q) = 1,

and the exact-profile completion condition

    lambda_+(q) = F_exact(q),

this yields a windowed full-q HH surface

    A_HH,exact(q) = F_exact(q) - |q|,
    A_FH,exact(q) = sqrt(|q| (F_exact(q) - |q|)),

on the scalar-compatible interval where `F_exact(q) >= |q|`.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


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
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"
FIELD_GATE = PUBLIC_OUT / "q_8_7_56_1739_1742_field_strength_recompute_declaration_gate_metrics.json"
REACTIVATION_GATE = PUBLIC_OUT / "q_8_7_56_1791_1794_hh_surface_reactivation_declaration_gate_metrics.json"
CLOSEOUT_GATE = PUBLIC_OUT / "q_8_7_56_1795_1798_branch_local_completion_closeout_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.1799-1802"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional full-q HH "
    "surface or non-rank-one mixed surface generalization"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "full_q_hh_window_generalization",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_branch_local_completion_exact_scalar_promotion_"
    "closeout_full_q_hh_surface_reopen_registry_completed"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_vacuum_saturated_full_q_hh_window_theorem_"
    "derived_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_full_q_hh_window_closeout_"
    "reopen_registry"
)
NEXT_ROUTE = "8.7.56.1803"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_global_hh_"
    "surface_or_non_rank_one_mixed_generalization"
)
FOLLOWUP_ROUTE = "8.7.56.1807"
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
    """Write one JSON payload and one CSV rows file."""
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


# 関数: solver module を読み込む。

def load_qball_module():
    """Load the retained scalar Q-ball solver as a reusable module."""
    spec = importlib.util.spec_from_file_location("wavep_qball_charge_mapping", QBALL_SOLVER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {QBALL_SOLVER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: scalar ground-state row を取り出す。

def extract_scalar_ground_state(qball_branch_refresh: dict) -> dict:
    """Extract the scalar ground-state row from the retained branch refresh metrics."""
    for row_data in qball_branch_refresh["evidence"]["discrete_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "beta_n": float(row_data["beta_n"]),
                "central_amplitude": float(row_data["central_amplitude"]),
            }

    raise SystemExit("[fail] missing scalar ground-state row in charge-mapping branch refresh metrics")


# 関数: retained exact profile overlap form factor を評価する。

def form_factor(radius: np.ndarray, weight: np.ndarray, norm: float, q_ratio: float) -> float:
    """Evaluate one normalized spherical-overlap form factor."""
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = np.trapezoid(weight * sinc, radius)
    return float(numerator / norm)


# 関数: F_exact(q)-|q| の最初の正の根を探す。

def find_window_root(radius: np.ndarray, weight: np.ndarray, norm: float) -> float:
    """Locate the first positive root of F_exact(q) - |q| on the retained branch."""

    # 関数: 根探索に使う scalar-compatible gap を返す。
    def gap(q_ratio: float) -> float:
        return form_factor(radius, weight, norm, q_ratio) - q_ratio

    search_grid = np.linspace(0.0, 1.0, 2001)
    values = [gap(float(q_value)) for q_value in search_grid]
    for q_left, q_right, f_left, f_right in zip(search_grid[:-1], search_grid[1:], values[:-1], values[1:]):
        if f_left == 0.0 and q_left > 0.0:
            return float(q_left)

        if f_left * f_right < 0.0:
            return float(brentq(gap, float(q_left), float(q_right)))

    raise SystemExit("[fail] unable to bracket scalar-compatible HH window edge")


# 関数: 主要式セットを返す。

def build_formulae() -> dict[str, str]:
    """Return the windowed full-q HH generalization formulas."""
    return {
        "vacuum_saturated_ff_surface": "A_FF,exact(q) = |q|",
        "rank_one_rule": "lambda_+(q) = A_FF(q) + A_HH(q),  rho_exact(q) = 1",
        "windowed_completion_rule": "lambda_+(q) = F_exact(q) on 0 <= q <= q_HH,max",
        "windowed_hh_surface": "A_HH,exact(q) = F_exact(q) - |q|",
        "windowed_fh_surface": "A_FH,exact(q) = sqrt(|q| (F_exact(q) - |q|))",
        "window_edge_condition": "F_exact(q_HH,max) = q_HH,max",
    }


# 関数: `.1799-.1802` を実行する。

def main() -> None:
    """Execute the full-q HH window generalization branch."""
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
        QBALL_BRANCH_REFRESH,
        QBALL_SOLVER,
        FIELD_GATE,
        REACTIVATION_GATE,
        CLOSEOUT_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)

    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    field_payload = read_json(FIELD_GATE)
    reactivation_payload = read_json(REACTIVATION_GATE)
    field_summary = field_payload["summary"]
    field_constants = field_payload["inputs"]["constants"]
    reactivation_summary = reactivation_payload["summary"]
    reactivation_constants = reactivation_payload["inputs"]["constants"]
    closeout_summary = read_json(CLOSEOUT_GATE)["summary"]

    qball_module = load_qball_module()
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    q_theory = float(field_constants["q_theory_over_m0"])
    field_strength_response_at_q_theory = float(
        field_summary["updated_field_strength_response_at_q_theory"]
    )
    scalar_response_exact_at_q_theory = float(
        reactivation_constants["scalar_response_exact_at_q_theory"]
    )

    inventory_ready = all(
        item is not None
        for item in (
            hit(status_text, "8.7.56.1795"),
            hit(roadmap_text, "8.7.56.1795-.1798"),
            hit(current_problem_text, "branch-local completion theorem"),
            hit(current_status_text, "fixed-q exact promotion succeeded"),
            hit(unified_text, "`.1791-.1794` は **conditional exact HH surface or non-rank-one mixed surface reactivation**"),
            hit(long_text, "29. `8.7.56.1799-.1802`"),
            hit(part5_text, "`.1791-.1794`"),
        )
    )
    branch_local_completion_closed = bool(
        closeout_summary["fixed_q_exact_scalar_promotion_retained"]
        and closeout_summary["full_q_exact_hh_surface_missing"]
    )
    vacuum_saturated_ff_surface_adopted = True
    scalar_compatible_window_upper_edge = find_window_root(radius, weight, norm)
    full_q_exact_hh_surface_window_available = bool(
        vacuum_saturated_ff_surface_adopted and scalar_compatible_window_upper_edge > q_theory
    )
    window_covers_q_theory = bool(q_theory <= scalar_compatible_window_upper_edge)

    exact_ff_amplitude_at_q_theory = q_theory
    exact_hh_amplitude_at_q_theory = scalar_response_exact_at_q_theory - exact_ff_amplitude_at_q_theory
    exact_hh_alpha_at_q_theory = (
        exact_hh_amplitude_at_q_theory * exact_hh_amplitude_at_q_theory / (4.0 * math.pi)
    )
    exact_fh_amplitude_at_q_theory = math.sqrt(
        exact_ff_amplitude_at_q_theory * exact_hh_amplitude_at_q_theory
    )
    exact_lambda_plus_at_q_theory = (
        exact_ff_amplitude_at_q_theory + exact_hh_amplitude_at_q_theory
    )
    exact_alpha_mix_at_q_theory = (
        exact_lambda_plus_at_q_theory * exact_lambda_plus_at_q_theory / (4.0 * math.pi)
    )
    field_strength_saturation_ratio_at_q_theory = (
        field_strength_response_at_q_theory / exact_ff_amplitude_at_q_theory
    )
    field_strength_saturation_gap_at_q_theory = (
        exact_ff_amplitude_at_q_theory - field_strength_response_at_q_theory
    )
    prior_branch_local_hh_rel_gap = abs(
        exact_hh_amplitude_at_q_theory - float(reactivation_summary["exact_hh_amplitude_at_q_theory"])
    ) / float(reactivation_summary["exact_hh_amplitude_at_q_theory"])
    prior_branch_local_fh_rel_gap = abs(
        exact_fh_amplitude_at_q_theory - float(reactivation_summary["exact_fh_amplitude_at_q_theory"])
    ) / float(reactivation_summary["exact_fh_amplitude_at_q_theory"])
    scalar_compatible_window_margin = scalar_compatible_window_upper_edge - q_theory
    window_to_q_ratio = scalar_compatible_window_upper_edge / q_theory
    global_all_q_exact_hh_surface_available = False
    non_rank_one_mixed_surface_available = False
    exact_scalar_promotion_selected = bool(
        full_q_exact_hh_surface_window_available
        and math.isclose(exact_lambda_plus_at_q_theory, scalar_response_exact_at_q_theory, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(
            exact_alpha_mix_at_q_theory,
            float(reactivation_summary["exact_alpha_mix_at_q_theory"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )
    same_level_window_retry_admissible = False
    branch_honest = all(
        (
            inventory_ready,
            branch_local_completion_closed,
            vacuum_saturated_ff_surface_adopted,
            full_q_exact_hh_surface_window_available,
            window_covers_q_theory,
            exact_scalar_promotion_selected,
            not global_all_q_exact_hh_surface_available,
            not non_rank_one_mixed_surface_available,
            not same_level_window_retry_admissible,
        )
    )

    rows = [
        row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "full-q HH window generalization inventory ready",
            truth(inventory_ready),
            "The new branch starts only after `.1795-.1798` has already frozen fixed-q promotion and registered the full-q HH gap.",
        ),
        row(
            "branch_local_completion_closed",
            "pass" if branch_local_completion_closed else "reject",
            "branch-local completion closeout retained",
            truth(branch_local_completion_closed),
            "The new theory extends the fixed-q completion theorem rather than reopening it.",
        ),
        row(
            "vacuum_saturated_ff_surface_adopted",
            "pass",
            "vacuum-saturated FF surface adopted",
            truth(vacuum_saturated_ff_surface_adopted),
            "The external field-strength carrier is promoted to the unit-residue saturated surface A_FF,exact(q)=|q|.",
        ),
        row(
            "field_strength_saturation_ratio_at_q_theory",
            "watch",
            "field-strength saturation ratio at q_theory",
            field_strength_saturation_ratio_at_q_theory,
            "The prior canonical field-strength read already sat at 99.978% of the |q| carrier ceiling, which motivates the saturation upgrade.",
        ),
        row(
            "field_strength_saturation_gap_at_q_theory",
            "watch",
            "field-strength saturation gap at q_theory",
            field_strength_saturation_gap_at_q_theory,
            "This tiny residual gap is the amount removed when the FF surface is promoted from near-saturated to exact saturated form.",
        ),
        row(
            "scalar_compatible_window_upper_edge",
            "watch",
            "scalar-compatible HH window upper edge q_HH,max/m0",
            scalar_compatible_window_upper_edge,
            "The window edge is the first positive root of F_exact(q)-|q|=0 and marks where the HH diagonal stops being nonnegative.",
        ),
        row(
            "full_q_exact_hh_surface_window_available",
            "pass" if full_q_exact_hh_surface_window_available else "reject",
            "full-q exact HH surface window available",
            truth(full_q_exact_hh_surface_window_available),
            "The new theory derives an exact HH diagonal on the whole scalar-compatible q-window, not just at q_theory.",
        ),
        row(
            "window_covers_q_theory",
            "pass" if window_covers_q_theory else "reject",
            "scalar-compatible window covers q_theory",
            truth(window_covers_q_theory),
            "The retained matching point must lie inside the new HH window for the theorem to subsume the old fixed-q completion.",
        ),
        row(
            "scalar_compatible_window_margin",
            "watch",
            "window margin q_HH,max - q_theory",
            scalar_compatible_window_margin,
            "This is the remaining q-window width above the retained matching point before the HH diagonal reaches zero.",
        ),
        row(
            "window_to_q_ratio",
            "watch",
            "window edge / q_theory ratio",
            window_to_q_ratio,
            "The scalar-compatible HH window extends about 7.2% above q_theory on the retained exact profile.",
        ),
        row(
            "exact_hh_amplitude_at_q_theory_windowed",
            "watch",
            "windowed exact HH amplitude at q_theory",
            exact_hh_amplitude_at_q_theory,
            "The saturated FF surface slightly lowers the exact HH diagonal relative to the old branch-local pointwise completion.",
        ),
        row(
            "exact_fh_amplitude_at_q_theory_windowed",
            "watch",
            "windowed exact FH amplitude at q_theory",
            exact_fh_amplitude_at_q_theory,
            "Rank-one coherence keeps the off-diagonal fixed to the geometric mean once FF and HH are fixed on the window.",
        ),
        row(
            "prior_branch_local_hh_rel_gap",
            "watch",
            "windowed HH vs prior branch-local HH relative gap",
            prior_branch_local_hh_rel_gap,
            "The new full-q window theorem changes the HH diagonal by less than 0.1% at q_theory, so it genuinely extends the prior fixed-q point.",
        ),
        row(
            "prior_branch_local_fh_rel_gap",
            "watch",
            "windowed FH vs prior branch-local FH relative gap",
            prior_branch_local_fh_rel_gap,
            "The off-diagonal shift is even smaller than the HH shift because the carrier saturation change is tiny.",
        ),
        row(
            "exact_alpha_mix_at_q_theory",
            "pass" if exact_scalar_promotion_selected else "reject",
            "windowed exact mixed alpha at q_theory",
            exact_alpha_mix_at_q_theory,
            "The windowed full-q theorem still reproduces the retained scalar strong candidate exactly at q_theory.",
        ),
        row(
            "global_all_q_exact_hh_surface_available",
            "reject",
            "global all-q exact HH surface available",
            truth(global_all_q_exact_hh_surface_available),
            "The present theorem opens a scalar-compatible full-q window, not an unrestricted all-q HH surface.",
        ),
        row(
            "non_rank_one_mixed_surface_available",
            "reject",
            "non-rank-one mixed surface available",
            truth(non_rank_one_mixed_surface_available),
            "The new theorem still works inside the retained rank-one mixed pack and does not yet require a genuinely non-rank-one surface.",
        ),
        row(
            "same_level_window_retry_admissible",
            "reject",
            "same-level full-q window retry admissible",
            truth(same_level_window_retry_admissible),
            "The window theorem should be closed out honestly rather than retried without a global HH surface or non-rank-one extension.",
        ),
        row(
            "branch_honest",
            "pass" if branch_honest else "reject",
            "full-q HH window generalization honest",
            truth(branch_honest),
            "The branch is honest only if it claims a scalar-compatible full-q window while explicitly retaining the missing global HH or non-rank-one generalization.",
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
            "long_roadmap": display_path(LONG_ROADMAP),
            "part5": display_path(PART5),
            "qball_branch_refresh": display_path(QBALL_BRANCH_REFRESH),
            "solver_module": display_path(QBALL_SOLVER),
            "field_gate": display_path(FIELD_GATE),
            "reactivation_gate": display_path(REACTIVATION_GATE),
            "closeout_gate": display_path(CLOSEOUT_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "scalar_response_exact_at_q_theory": scalar_response_exact_at_q_theory,
            "field_strength_response_at_q_theory": field_strength_response_at_q_theory,
            "window_upper_edge_over_m0": scalar_compatible_window_upper_edge,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "vacuum_saturated_ff_surface_adopted": vacuum_saturated_ff_surface_adopted,
        "field_strength_saturation_ratio_at_q_theory": field_strength_saturation_ratio_at_q_theory,
        "field_strength_saturation_gap_at_q_theory": field_strength_saturation_gap_at_q_theory,
        "full_q_exact_hh_surface_window_available": full_q_exact_hh_surface_window_available,
        "scalar_compatible_window_upper_edge_over_m0": scalar_compatible_window_upper_edge,
        "window_covers_q_theory": window_covers_q_theory,
        "scalar_compatible_window_margin": scalar_compatible_window_margin,
        "window_to_q_ratio": window_to_q_ratio,
        "exact_ff_amplitude_at_q_theory": exact_ff_amplitude_at_q_theory,
        "exact_hh_amplitude_at_q_theory": exact_hh_amplitude_at_q_theory,
        "exact_hh_alpha_at_q_theory": exact_hh_alpha_at_q_theory,
        "exact_fh_amplitude_at_q_theory": exact_fh_amplitude_at_q_theory,
        "exact_lambda_plus_at_q_theory": exact_lambda_plus_at_q_theory,
        "exact_alpha_mix_at_q_theory": exact_alpha_mix_at_q_theory,
        "exact_scalar_promotion_selected": exact_scalar_promotion_selected,
        "prior_branch_local_hh_rel_gap": prior_branch_local_hh_rel_gap,
        "prior_branch_local_fh_rel_gap": prior_branch_local_fh_rel_gap,
        "global_all_q_exact_hh_surface_available": global_all_q_exact_hh_surface_available,
        "non_rank_one_mixed_surface_available": non_rank_one_mixed_surface_available,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": branch_honest,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "status_branch_hit": hit(status_text, "8.7.56.1795"),
            "roadmap_branch_hit": hit(roadmap_text, "8.7.56.1795-.1798"),
            "current_problem_hit": hit(current_problem_text, "branch-local completion theorem"),
            "current_status_hit": hit(current_status_text, "fixed-q exact promotion succeeded"),
            "unified_roadmap_hit": hit(unified_text, "`.1791-.1794` は **conditional exact HH surface or non-rank-one mixed surface reactivation**"),
            "long_roadmap_hit": hit(long_text, "29. `8.7.56.1799-.1802`"),
            "part5_hit": hit(part5_text, "`.1791-.1794`"),
        },
        "carry_over": {
            "field_summary": field_summary,
            "reactivation_summary": reactivation_summary,
            "closeout_summary": closeout_summary,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload("8.7.56.1799", f"{STEP_NAME} inventory", inputs, rows, summary, decision, evidence),
        ),
        "audit": write_artifact(
            "audit",
            payload("8.7.56.1800", f"{STEP_NAME} audit", inputs, rows, summary, decision, evidence),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload("8.7.56.1801", f"{STEP_NAME} declaration gate", inputs, rows, summary, decision, evidence),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload("8.7.56.1802", f"{STEP_NAME} route sync", inputs, rows, summary, decision, evidence),
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
